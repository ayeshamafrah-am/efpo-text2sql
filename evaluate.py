import json, sqlite3, sys, time
from collections import defaultdict

def run_sql(db_path, sql, timeout=30):
    con = sqlite3.connect(db_path); start = time.time()
    def guard():
        if time.time() - start > timeout: return 1
    con.set_progress_handler(guard, 10000)
    try:
        rows = con.execute(sql).fetchall()
        return "ok", set(rows)
    except Exception as e:
        return "error", str(e)
    finally:
        con.close()

def evaluate(data_file, pred_file):
    data = [json.loads(l) for l in open(data_file)]
    preds = [json.loads(l)["pred_sql"] for l in open(pred_file)]
    assert len(data) == len(preds), "pred/data length mismatch"
    tot = defaultdict(int); correct = defaultdict(int); errors = 0
    for ex, p in zip(data, preds):
        d = ex["difficulty"]; tot[d] += 1; tot["ALL"] += 1
        gs, gold = run_sql(ex["db_path"], ex["sql"])
        ps, pred = run_sql(ex["db_path"], p)
        if ps == "error": errors += 1
        if gs == "ok" and ps == "ok" and gold == pred:
            correct[d] += 1; correct["ALL"] += 1
    print(f"\nExecution accuracy  ({data_file} vs {pred_file})")
    for d in list(tot):
        print(f"  {d:12s} {correct[d]:5d}/{tot[d]:5d}  = {100*correct[d]/tot[d]:5.1f}%")
    print(f"  SQL errors in predictions: {errors}")

if __name__ == "__main__":
    evaluate(sys.argv[1], sys.argv[2])
