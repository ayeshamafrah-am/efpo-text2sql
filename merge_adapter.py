import sys, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
MODEL = "Qwen/Qwen2.5-Coder-3B-Instruct"; adapter, out = sys.argv[1:3]
base = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16, device_map="cpu")
m = PeftModel.from_pretrained(base, adapter).merge_and_unload()
m.save_pretrained(out); AutoTokenizer.from_pretrained(MODEL).save_pretrained(out)
print("merged ->", out)
