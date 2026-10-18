import json, sys, torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig

MODEL = "Qwen/Qwen2.5-Coder-3B-Instruct"
train_file = sys.argv[1]; out_dir = sys.argv[2]

def build_prompt(ex):
    ev = f"\nHint: {ex['evidence']}" if ex["evidence"] else ""
    return (f"Database schema:\n{ex['schema']}\n{ev}\n"
            f"Question: {ex['question']}\n"
            f"Write one SQLite query that answers the question. Return only the SQL.")

rows = [json.loads(l) for l in open(train_file)]
ds = Dataset.from_list([{
    "prompt": [{"role": "user", "content": build_prompt(ex)}],
    "completion": [{"role": "assistant", "content": f"```sql\n{ex['sql']}\n```"}],
} for ex in rows])
print("train examples:", len(ds))

tok = AutoTokenizer.from_pretrained(MODEL)
bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                         bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
model = AutoModelForCausalLM.from_pretrained(MODEL, quantization_config=bnb, device_map="cuda")
model = prepare_model_for_kbit_training(model)

lora = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, task_type="CAUSAL_LM",
                  target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"])

cfg = SFTConfig(
    output_dir=out_dir, num_train_epochs=2,
    per_device_train_batch_size=2, gradient_accumulation_steps=8,
    learning_rate=2e-4, lr_scheduler_type="cosine", warmup_steps=30,
    bf16=True, gradient_checkpointing=True, max_length=3072,
    logging_steps=10, save_strategy="epoch", report_to="none", seed=42,
)
trainer = SFTTrainer(model=model, args=cfg, train_dataset=ds, processing_class=tok, peft_config=lora)
trainer.train()
trainer.save_model(out_dir)
print("Peak VRAM:", round(torch.cuda.max_memory_allocated()/1e9, 2), "GB")
