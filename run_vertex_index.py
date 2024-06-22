from yaml import safe_load

from models import ModelManager


"""
Download corpus.jsonl by running
>>> wget https://huggingface.co/datasets/mteb/nq/resolve/main/corpus.jsonl
"""

MODEL_META_PATH = "model_meta.yml"
# Debugging
# MODEL_META_PATH = "model_meta_debug.yml"
with open(MODEL_META_PATH, 'r', encoding='utf-8') as f:
    model_meta = safe_load(f)
models = ModelManager(model_meta, use_gcp_index=True)


## Test with a small model
model_name = "sentence-transformers/all-MiniLM-L6-v2"
models.retrieve(query="What is this code telling me?", model_name=model_name)