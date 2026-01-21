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

## Adding a Model

1. Add the model to [mteb](https://github.com/embeddings-benchmark/mteb/tree/main/mteb/models) via PR.
2. Add the model to this repo's `models.py` & `models_meta.yml` via PR.
3. Test it in the Clustering and STS interfaces at least. Ideally, also retrieval.
4. We will then consider adding it to the arena. Please note that keeping a model in the arena is costly to us as (1) we need to keep constant GPUs running with it in memory (2) we need to maintain retrieval indices for it. So we will focus on models that are most useful to the community. However, if you are willing to sponsor the hosting of your model, we may be able to add it faster.

## Results

Results are auto-saved to [mteb/arena-results](https://huggingface.co/datasets/mteb/arena-results).

## Personal Leaderboards

We have introduced a new feature that allows users to have their own personal leaderboards. This feature enables users to track their individual preferences and votes.

### User Authentication

To access your personal leaderboard, you need to log in using your Hugging Face account. The authentication is handled via OAuth.

### Instructions

1. Log in to your Hugging Face account.
2. Navigate to the "Personal Leaderboard" tab in the UI.
3. View your personal rankings based on your votes.