import json, sys, re, torch, time
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

MODEL = sys.argv[1]; data_file = sys.argv[2]; out_file = sys.argv[3]
BATCH = 8

def build_prompt(ex):
    ev = f"\nHint: {ex['evidence']}" if ex["evidence"] else ""
    return (f"Database schema:\n{ex['schema']}\n{ev}\n"
            f"Question: {ex['question']}\n"
            f"Write one SQLite query that answers the question. Return only the SQL.")

def extract_sql(text):
    m = re.search(r"```(?:sql)?\s*(.*?)```", text, re.S)
    sql = m.group(1) if m else text
    return sql.strip().split(";")[0].strip()

tok = AutoTokenizer.from_pretrained(MODEL); tok.padding_side = "left"
bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)
model = AutoModelForCausalLM.from_pretrained(MODEL, quantization_config=bnb, device_map="cuda")
if len(sys.argv) > 4:
    from peft import PeftModel
    model = PeftModel.from_pretrained(model, sys.argv[4]); model.eval()

data = [json.loads(l) for l in open(data_file)]
t0 = time.time()
with open(out_file, "w") as f:
    for i in range(0, len(data), BATCH):
        batch = data[i:i+BATCH]
        msgs = [[{"role": "user", "content": build_prompt(ex)}] for ex in batch]
        inp = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt",
                                      return_dict=True, padding=True).to("cuda")
        with torch.no_grad():
            out = model.generate(**inp, max_new_tokens=256, do_sample=False)
        for ex, o in zip(batch, out):
            text = tok.decode(o[inp["input_ids"].shape[1]:], skip_special_tokens=True)
            f.write(json.dumps({"db_id": ex["db_id"], "pred_sql": extract_sql(text)}) + "\n")
        print(f"{i+len(batch)}/{len(data)}  {time.time()-t0:.0f}s", flush=True)
print("Peak VRAM:", round(torch.cuda.max_memory_allocated()/1e9, 2), "GB")
