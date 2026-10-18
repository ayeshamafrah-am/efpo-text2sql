import json, re, random, sys
from collections import Counter
random.seed(42)
MAX_REJ = 2   # rejected candidates per question

def build_prompt(ex):
    ev = f"\nHint: {ex['evidence']}" if ex["evidence"] else ""
    return (f"Database schema:\n{ex['schema']}\n{ev}\n"
            f"Question: {ex['question']}\n"
            f"Write one SQLite query that answers the question. Return only the SQL.")

def feats(sql):
    s = sql.upper()
    return {"join": " JOIN " in s,
            "nested": s.count("SELECT") > 1 and not re.search(r"\b(UNION|INTERSECT|EXCEPT)\b", s),
            "setop": bool(re.search(r"\b(UNION|INTERSECT|EXCEPT)\b", s)),
            "group": "GROUP BY" in s or "HAVING" in s,
            "order": "ORDER BY" in s or "LIMIT" in s,
            "agg": bool(re.search(r"\b(COUNT|SUM|AVG|MAX|MIN)\s*\(", s))}

def tags(chosen, rejected):
    fc, fr = feats(chosen), feats(rejected)
    t = [k for k in fc if fc[k] != fr[k]]
    return t if t else ["surface"]

pairs = []
for cand_file, data_file in [("cands/spider_train_sftmix_n8.jsonl", "data/unified/spider_train.jsonl"),
                             ("cands/bird_train4k_sftmix_n8.jsonl", "data/unified/bird_train_4k.jsonl")]:
    data = [json.loads(l) for l in open(data_file)]
    for r in map(json.loads, open(cand_file)):
        if not r["gold_ok"]: continue
        ex = data[r["idx"]]
        good = [c["sql"] for c in r["cands"] if c["correct"]]
        bad  = [c for c in r["cands"] if not c["correct"] and c["sql"]]
        if not bad: continue
        if good: chosen, ptype = random.choice(good), "self"      # model's own correct answer
        else:    chosen, ptype = ex["sql"], "gold"                # all samples wrong -> gold as chosen
        random.shuffle(bad)
        for c in bad[:MAX_REJ]:
            if c["sql"].strip().upper() == chosen.strip().upper(): continue
            pairs.append({"prompt": [{"role": "user", "content": build_prompt(ex)}],
                          "chosen": [{"role": "assistant", "content": f"```sql\n{chosen}\n```"}],
                          "rejected": [{"role": "assistant", "content": f"```sql\n{c['sql']}\n```"}],
                          "source": ex["source"], "difficulty": ex["difficulty"], "pair_type": ptype,
                          "rej_status": c["status"], "tags": tags(chosen, c["sql"])})

random.shuffle(pairs)
n_eval = int(0.05 * len(pairs))
with open("data/unified/pairs_train.jsonl", "w") as f:
    for p in pairs[n_eval:]: f.write(json.dumps(p) + "\n")
with open("data/unified/pairs_eval.jsonl", "w") as f:
    for p in pairs[:n_eval]: f.write(json.dumps(p) + "\n")

print("total pairs:", len(pairs), " train:", len(pairs)-n_eval, " eval:", n_eval)
print("by source:   ", dict(Counter(p["source"] for p in pairs)))
print("by pair_type:", dict(Counter(p["pair_type"] for p in pairs)))
print("by rej_status:", dict(Counter(p["rej_status"] for p in pairs)))
print("by tag:      ", dict(Counter(t for p in pairs for t in p["tags"])))
print("by difficulty:", dict(Counter(p["difficulty"] for p in pairs)))
