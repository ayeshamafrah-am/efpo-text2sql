import json, sys, re, os, torch, time
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from evaluate import run_sql

MODEL, ADAPTER, data_file, out_file = sys.argv[1:5]
N = int(sys.argv[5]) if len(sys.argv) > 5 else 8
BATCH = 1  # prompts per batch (x N samples each)

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
model = PeftModel.from_pretrained(model, ADAPTER); model.eval()

data = [json.loads(l) for l in open(data_file)]
done = set()
if os.path.exists(out_file):
    done = {json.loads(l)["idx"] for l in open(out_file)}
    print(f"resuming: {len(done)} already done")
todo = [i for i in range(len(data)) if i not in done]

t0 = time.time(); usable = 0; processed = 0
with open(out_file, "a") as f:
    for b in range(0, len(todo), BATCH):
        idxs = todo[b:b+BATCH]; batch = [data[i] for i in idxs]
        msgs = [[{"role": "user", "content": build_prompt(ex)}] for ex in batch]
        inp = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt",
                                      return_dict=True, padding=True).to("cuda")
        with torch.no_grad():
            out = model.generate(**inp, max_new_tokens=256, do_sample=True, temperature=0.8,
                                 top_p=0.95, num_return_sequences=N)
        L = inp["input_ids"].shape[1]
        for j, (i, ex) in enumerate(zip(idxs, batch)):
            gs, gold = run_sql(ex["db_path"], ex["sql"])
            cands = []
            seen = set()
            for k in range(N):
                sql = extract_sql(tok.decode(out[j*N+k][L:], skip_special_tokens=True))
                if sql in seen: continue
                seen.add(sql)
                ps, pred = run_sql(ex["db_path"], sql)
                cands.append({"sql": sql, "status": ps,
                              "correct": bool(gs == "ok" and ps == "ok" and pred == gold)})
            nc = sum(c["correct"] for c in cands)
            if 0 < nc < len(cands): usable += 1
            processed += 1
            f.write(json.dumps({"idx": i, "source": ex["source"], "db_id": ex["db_id"],
                                "difficulty": ex["difficulty"], "gold_ok": gs == "ok",
                                "n_cands": len(cands), "n_correct": nc, "cands": cands}) + "\n")
        f.flush()
        if processed % 20 == 0:
            print(f"{len(done)+processed}/{len(data)}  usable so far {usable}/{processed}  {time.time()-t0:.0f}s", flush=True)
print("Peak VRAM:", round(torch.cuda.max_memory_allocated()/1e9, 2), "GB")
