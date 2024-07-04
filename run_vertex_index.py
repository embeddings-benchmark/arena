from yaml import safe_load

from models import ModelManager

"""
Script for testing the VertexIndex code and setup the vector search index.

Download corpus.jsonl by running
>>> wget https://huggingface.co/datasets/mteb/nq/resolve/main/corpus.jsonl
"""

MODEL_META_PATH = "model_meta.yml"
with open(MODEL_META_PATH, 'r', encoding='utf-8') as f:
    model_meta = safe_load(f)

## Enable GCP Vertex index
models = ModelManager(model_meta, use_gcp_index=True)


## Test with a small model
model_name = "GritLM/GritLM-7B"
print(models.retrieve(query="What is this code telling me?", model_name=model_name))