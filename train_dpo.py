import json, sys, torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, prepare_model_for_kbit_training
from trl import DPOTrainer, DPOConfig

SFT_MERGED, train_file, out_dir = sys.argv[1:4]

def load(path):
    rows = [json.loads(l) for l in open(path)]
    return Dataset.from_list([{"prompt": r["prompt"], "chosen": r["chosen"], "rejected": r["rejected"]} for r in rows])
train_ds = load(train_file); eval_ds = load("data/unified/pairs_eval.jsonl")
print("train pairs:", len(train_ds), " eval pairs:", len(eval_ds))

tok = AutoTokenizer.from_pretrained(SFT_MERGED)
bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                         bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
model = AutoModelForCausalLM.from_pretrained(SFT_MERGED, quantization_config=bnb, device_map="cuda")
model = prepare_model_for_kbit_training(model)
lora = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, task_type="CAUSAL_LM",
                  target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"])

cfg = DPOConfig(
    output_dir=out_dir, num_train_epochs=1, beta=0.1,
    per_device_train_batch_size=1, gradient_accumulation_steps=16,
    learning_rate=2e-5, lr_scheduler_type="cosine", warmup_steps=20,
    bf16=True, gradient_checkpointing=True, max_length=3072,
    logging_steps=5, eval_strategy="steps", eval_steps=100, per_device_eval_batch_size=1,
    save_strategy="epoch", report_to="none", seed=42,
)
trainer = DPOTrainer(model=model, ref_model=None, args=cfg, train_dataset=train_ds,
                     eval_dataset=eval_ds, processing_class=tok, peft_config=lora)
trainer.train()
trainer.save_model(out_dir)
print("Peak VRAM:", round(torch.cuda.max_memory_allocated()/1e9, 2), "GB")
