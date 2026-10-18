import torch, time
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

MODEL = "Qwen/Qwen2.5-Coder-3B-Instruct"

print("GPU:", torch.cuda.get_device_name(0))
bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                         bnb_4bit_compute_dtype=torch.bfloat16)
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, quantization_config=bnb, device_map="cuda")

used = torch.cuda.memory_allocated() / 1e9
print(f"Model loaded. VRAM used: {used:.2f} GB  (free for training ~ {12 - used:.1f} GB)")

schema = "CREATE TABLE employees(id INT, name TEXT, dept TEXT, salary INT);"
question = "Which department has the highest average salary?"
msgs = [{"role": "user", "content": f"{schema}\n\nQuestion: {question}\nReturn only the SQL."}]
inp = tok.apply_chat_template(msgs, add_generation_prompt=True,
                              return_tensors="pt", return_dict=True).to("cuda")

t = time.time()
out = model.generate(**inp, max_new_tokens=80, do_sample=False)
print("Generated in", round(time.time() - t, 1), "s:")
print(tok.decode(out[0][inp["input_ids"].shape[1]:], skip_special_tokens=True))
print("Peak VRAM:", round(torch.cuda.max_memory_allocated() / 1e9, 2), "GB")
