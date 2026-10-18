# EFPO: Execution-Feedback Preference Optimization for Small-Model Text-to-SQL

Code, logs and predictions for the paper *Execution-Feedback Preference Optimization for Small-Model Text-to-SQL*.
All experiments were run on one laptop GPU (NVIDIA RTX 5070 Ti, 12 GB) with Qwen2.5-Coder-3B-Instruct in 4-bit.

Model adapters (SFT-mixed and EFPO / DPO-v3) are hosted on Hugging Face: see the link in the paper, Section IX.

## Pipeline

| Step | Script | Output |
|---|---|---|
| Data unification (Spider + BIRD) | `prepare_data.py` | `data/unified/*.jsonl` |
| Greedy inference | `infer.py` | `preds/*.jsonl` |
| Execution-accuracy evaluator | `evaluate.py` | printed table |
| Supervised fine-tuning (QLoRA) | `train_sft.py` | `ckpt/sft_3b_*` |
| Candidate sampling + execution labelling | `sample_candidates.py` | `cands/*.jsonl` |
| Pair construction + clause tagging | `build_pairs.py` | `pairs/*.jsonl` |
| Adapter merge | `merge_adapter.py` | `ckpt/sft_3b_mixed_merged` |
| Regularized DPO (EFPO = v3) | `train_dpo_v3.py` (v1 `train_dpo.py`, v2 `train_dpo_v2.py`) | `ckpt/dpo_3b_*` |

`results/EXPERIMENT_LOG.md` records every run with date, configuration, wall-clock, peak VRAM and per-tier accuracy; each table in the paper maps to entries in that log. `results/analysis_spider_v3_vs_sft.txt` is the error analysis.

## Reproduce

```bash
python3 -m venv venv && source venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
git clone https://github.com/taoyds/test-suite-sql-eval.git evalsuite
# Download Spider (official zip, SHA-256 00636695dabed6b5f4b8328a16b13e069a2f16591d5efcce57660669c85b121b)
# into data/spider_data and BIRD dev/train into data/bird, then:
python prepare_data.py
python train_sft.py data/unified/mixed_train.jsonl ckpt/sft_3b_mixed
python sample_candidates.py Qwen/Qwen2.5-Coder-3B-Instruct ckpt/sft_3b_mixed data/unified/spider_train.jsonl cands/spider_train_sftmix_n8.jsonl 8
python build_pairs.py
python merge_adapter.py ckpt/sft_3b_mixed ckpt/sft_3b_mixed_merged
python train_dpo_v3.py ckpt/sft_3b_mixed_merged pairs/pairs_train_weighted.jsonl ckpt/dpo_3b_v3
python infer.py ckpt/sft_3b_mixed_merged data/unified/bird_dev.jsonl preds/dpov3_3b_bird_dev.jsonl ckpt/dpo_3b_v3
python evaluate.py data/unified/bird_dev.jsonl preds/dpov3_3b_bird_dev.jsonl
```

Datasets are not redistributed here; obtain Spider and BIRD from their official sources.

## Results (development sets, greedy decoding)

| Model | Spider dev EX | BIRD dev EX | BIRD errors |
|---|---|---|---|
| Zero-shot | 62.7 | 36.8 | 476 |
| SFT-Spider | 76.4 | 35.7 | 389 |
| SFT-mixed | 77.4 | 47.0 | 171 |
| EFPO (ours) | 74.6 | 47.7 | 159 |

## License

Code: MIT. Spider and BIRD retain their original licenses.
