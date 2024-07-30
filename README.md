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

## Run

`python app.py`

If you'd like to debug the app, set `DEBUG=True` in the beginning of `app.py`, which will use fewer models and can be easily run locally on CPUs. Also you may want to set `GCP_INDEX=False` and just use the local indices for the two debug models.

Some models require API keys which you can set as environment variables, e.g.
`VOYAGE_API_KEY=YOUR_KEY HF_TOKEN=YOUR_KEY OPENAI_API_KEY=YOUR_KEY CO_API_KEY=YOUR_KEY python app.py`

## Results

Results are auto-saved to [mteb/arena-results](https://huggingface.co/datasets/mteb/arena-results).
