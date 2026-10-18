## 2026-10-14 — Zero-shot baseline, Qwen2.5-Coder-3B-Instruct (4-bit NF4), greedy, Spider dev
EX: ALL 62.7 | easy 85.1 | medium 61.0 | hard 54.0 | extra 42.8 | exec errors 157/1034
Log: results/zeroshot_3b_spider_dev.log  Preds: preds/zeroshot_3b_spider_dev.jsonl
## 2026-10-14 — Zero-shot baseline, Qwen2.5-Coder-3B-Instruct (4-bit NF4), greedy, BIRD dev (with evidence)
EX: ALL 36.8 | simple 44.6 | moderate 26.1 | challenging 20.7 | exec errors 476/1534
Log: results/zeroshot_3b_bird_dev.log  Preds: preds/zeroshot_3b_bird_dev.jsonl
## 2026-10-14 — SFT (QLoRA r=16, 2 epochs, Spider train only), Qwen2.5-Coder-3B, greedy
Train: 866 steps, 2h23m, final loss 0.04, peak VRAM 5.12 GB. Log: results/sft_3b_spider_train.log
Spider dev EX: ALL 76.4 | easy 92.7 | medium 77.1 | hard 73.6 | extra 53.0 | errors 45   (zero-shot was 62.7)
BIRD dev  EX: ALL 35.7 | simple 45.1 | moderate 22.6 | challenging 17.2 | errors 389  (zero-shot was 36.8)
Finding: Spider-only SFT does not transfer to BIRD -> motivates multi-dataset + execution-feedback training.
## 2026-10-15 — SFT-mixed (QLoRA r=16, 2 epochs, Spider train + BIRD train, max_len 3072), Qwen2.5-Coder-3B, greedy
Train: 1996 steps, 7h50m, ~460 examples dropped for exceeding 3072 tokens, final loss ~0.06, peak VRAM 6.93 GB
Spider dev EX: ALL 77.4 | easy 94.0 | medium 78.9 | hard 72.4 | extra 53.6 | errors 36
BIRD dev  EX: ALL 47.0 | simple 54.7 | moderate 35.1 | challenging 35.9 | errors 171
Finding: mixed-dataset SFT generalizes (BIRD +11.3 over Spider-only SFT) without hurting Spider. -> official SFT baseline.
## 2026-10-17 — Control: merged SFT-mixed (bf16 merge, requantized 4-bit), no DPO, BIRD dev
EX: ALL 46.3 | simple 54.1 | moderate 36.0 | challenging 29.7 | errors 195  (adapter version was 47.0) -> merge OK, DPO-v1 recipe at fault
## 2026-10-17 — DPO-v2 (beta 0.2, lr 5e-6, sigmoid+sft loss, 1 epoch) on merged SFT-mixed
Train: 3h57m, eval rewards/chosen -0.004 (fixed), rewards/rejected -0.065, acc 0.656, margins 0.06
Spider dev EX: ALL 78.1 | easy 94.0 | medium 79.4 | hard 75.3 | extra 54.2 | errors 36   (SFT-mixed 77.4)
BIRD dev  EX: ALL 46.5 | simple 54.2 | moderate 36.2 | challenging 30.3 | errors 188 (control 46.3)
Result: no significant change (within noise). Update too conservative; hard tiers and errors move in the right direction.
## 2026-10-17 — DPO-v3 (beta 0.1, lr 2e-5, sigmoid+sft loss, 1 epoch) on merged SFT-mixed  [= v1 + SFT term]
Train: 482 steps, 3h57m, peak 6.77 GB. Final eval: rewards/chosen -0.18, rewards/rejected -0.68, acc 0.691, margins 0.50
(v1 had chosen -0.62; v2 had margins only 0.06) -> strongest margin with chosen likelihood mostly preserved.
NOTE: train log overwritten by accidental rerun; metrics transcribed from terminal.
## 2026-10-17 — DPO-v3 eval
Spider dev EX: ALL 74.6 | easy 92.3 | medium 74.4 | hard 70.7 | extra 52.4 | errors 33   (SFT-mixed 77.4)
BIRD dev  EX: ALL 47.7 | simple 55.6 | moderate 36.6 | challenging 32.4 | errors 159 (control 46.3; best BIRD so far)
Result: BIRD +1.4 overall, challenging +2.7, errors -18%; Spider -2.8. Trade-off: strong update shifts toward BIRD (66% of pairs).
## 2026-10-18 — DPO-v4 (= v3 settings, source-balanced pairs, Spider x2 -> 10,318 pairs), 646 steps, 5h06m
Final eval: rewards/chosen -0.22, rejected -0.89, acc 0.68, margins 0.68
Spider dev EX: ALL 73.2 | easy 91.5 | medium 74.2 | hard 70.1 | extra 46.4 | errors 37
BIRD dev  EX: ALL 47.3 | simple 55.4 | moderate 36.4 | challenging 31.0 | errors 154
Result: balancing did NOT recover Spider (73.2 < v3 74.6). Hypothesis "BIRD skew causes Spider drop" rejected. Spider loss tracks update strength, not source mix.
Error analysis v3 vs SFT (Spider dev): losses are schema-fidelity errors (wrong/hallucinated column names, spurious joins, column reorder, truncated literals), not logic. Hypothesis: 70% surface pairs inject noisy token-level signal. -> v5: structural pairs only.
## 2026-10-18 — DPO-v5 (v3 settings, structural pairs only: 1854 pairs, 111 steps, 1h01m), margins 0.035
Spider dev EX: ALL 77.2 | easy 92.7 | medium 78.0 | hard 75.9 | extra 53.0 | errors 40
BIRD dev  EX: ALL 46.3 | simple 53.7 | moderate 36.6 | challenging 30.3 | errors 203
Result: Spider preserved (confirms surface pairs cause the Spider drop) but too few pairs to move BIRD. ~= SFT.
EXPERIMENTS CLOSED. Method for paper = DPO-v3. Remaining: Spider test eval (SFT-mixed merged, DPO-v3).
## 2026-10-18 — Spider TEST (2147 q, held out, evaluated once)
SFT-mixed (merged): ALL 76.6 | easy 90.6 | medium 76.9 | hard 70.6 | extra 65.0 | errors 72
DPO-v3:             ALL 75.6 | easy 90.6 | medium 76.5 | hard 70.2 | extra 60.8 | errors 82
Result: no gain on Spider test (-1.0, within noise except extra -4.2). Consistent with dev findings.
