import json, sqlite3, os, sys
sys.path.insert(0, "evalsuite")
from evaluation import Evaluator
ev = Evaluator()

def schema_from_sqlite(db_path):
    con = sqlite3.connect(db_path)
    rows = con.execute("SELECT sql FROM sqlite_master WHERE type='table' AND sql IS NOT NULL").fetchall()
    con.close()
    return "\n".join(r[0] for r in rows)

def spider(split_file, db_root, source):
    out = []
    for ex in json.load(open(split_file)):
        db = ex["db_id"]
        db_path = f"{db_root}/{db}/{db}.sqlite"
        try: diff = ev.eval_hardness(ex["sql"])
        except Exception: diff = "unknown"
        out.append(dict(source=source, db_id=db, db_path=db_path,
                        question=ex["question"], evidence="",
                        sql=ex["query"], difficulty=diff,
                        schema=schema_from_sqlite(db_path)))
    return out

def bird(split_file, db_root, source):
    out = []
    for ex in json.load(open(split_file)):
        db = ex["db_id"]
        db_path = f"{db_root}/{db}/{db}.sqlite"
        out.append(dict(source=source, db_id=db, db_path=db_path,
                        question=ex["question"], evidence=ex.get("evidence",""),
                        sql=ex["SQL"], difficulty=ex.get("difficulty","unlabeled"),
                        schema=schema_from_sqlite(db_path)))
    return out

S = "data/spider_data"; B = "data/bird/dev_20240627"
os.makedirs("data/unified", exist_ok=True)
sets = {
  "spider_train": spider(f"{S}/train_spider.json", f"{S}/database", "spider"),
  "spider_dev":   spider(f"{S}/dev.json",          f"{S}/database", "spider"),
  "spider_test":  spider(f"{S}/test.json",         f"{S}/test_database", "spider"),
  "bird_dev":     bird(f"{B}/dev.json",            f"{B}/dev_databases", "bird"),
  "bird_train":   bird("data/bird/train/train.json", "data/bird/train/train_databases", "bird"),
}
for name, rows in sets.items():
    with open(f"data/unified/{name}.jsonl", "w") as f:
        for r in rows: f.write(json.dumps(r) + "\n")
    from collections import Counter
    print(f"{name:13s} {len(rows):5d}  difficulty:", dict(Counter(r['difficulty'] for r in rows)))
