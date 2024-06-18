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
3. `wget https://huggingface.co/datasets/BeIR/nq/resolve/main/corpus.jsonl.gz`
4. `gunzip corpus.jsonl.gz`
5. Maybe allow creating index here

## Run

`python app.py`

## Results

Results are auto-saved to [mteb/arena-results](https://huggingface.co/datasets/mteb/arena-results).
