"""Script for testing the VertexIndex code and setup the vector search index."""
import sys
from yaml import safe_load
from models import ModelManager

MODEL_META_PATH = "model_meta.yml"
with open(MODEL_META_PATH, 'r', encoding='utf-8') as f:
    model_meta = safe_load(f)

## Enable GCP Vertex index
models = ModelManager(model_meta, use_gcp_index=True)

model_name = sys.argv[1] # "intfloat/multilingual-e5-small"
corpus = sys.argv[2] # "wikipedia"

print(models.retrieve(query="Where is Japan?", model_name=model_name, corpus=corpus))