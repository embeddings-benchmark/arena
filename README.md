---
title: MTEB Arena
emoji: ⚔️
colorFrom: indigo
colorTo: blue
sdk: gradio
sdk_version: 4.21.0
app_file: app.py
pinned: false
tags:
- arena
- leaderboard
---

# MTEB Arena

## Setup

1. Install PyTorch according to your system
2. `pip install -r requirements.txt`
3. Retrieval: Cache indices so retrieval will be faster
3.1 `wget https://huggingface.co/datasets/BeIR/nq/resolve/main/corpus.jsonl.gz`
3.2 `gunzip corpus.jsonl.gz`
3.3 Maybe allow creating index here
4. Clustering: TODO

## Run

`python app.py`

# python gritlm/rag/eval.py --model_name_or_path GritLM/gritlm-7b --eval_data gritlm/rag/nq_data/test.jsonl --passages corpus.jsonl --save_index_path index_nq
