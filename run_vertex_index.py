from yaml import safe_load

from models import ModelManager

"""
Script for testing the VertexIndex code and setup the vector search index.
Note that batch updating the index will take a long time. Consider switching 
to stream update instead (costs more).

Download corpus.jsonl by running
>>> wget https://huggingface.co/datasets/mteb/nq/resolve/main/corpus.jsonl
"""

MODEL_META_PATH = "model_meta.yml"
with open(MODEL_META_PATH, 'r', encoding='utf-8') as f:
    model_meta = safe_load(f)

## Enable GCP Vertex index
models = ModelManager(model_meta, use_gcp_index=True)


## Test with a small model
model_name = "sentence-transformers/all-MiniLM-L6-v2"
print(models.retrieve(query="What is this code telling me?", model_name=model_name))