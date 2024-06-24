from yaml import safe_load
import random
import json
import yaml
from models import ModelManager

random.seed(42)

MODEL_META_PATH = "model_meta_debug.yml"
with open(MODEL_META_PATH, 'r', encoding='utf-8') as f:
    model_meta = safe_load(f)

## Get a small query set
from datasets import load_dataset
all_queries = load_dataset("mteb/nq", "queries")["queries"]["text"]
queries = random.choices(all_queries, k=10)

## Test with a small model
model_name = "sentence-transformers/all-MiniLM-L6-v2"

results = dict()
for use_gcp in [False, True]:
    index = "gcp" if use_gcp else "local"
    models = ModelManager(model_meta, use_gcp_index=use_gcp)
    for q in queries:
        docs = models.retrieve(query=q, model_name=model_name)
        if results.get(q):
            results[q].update({index: docs[0][1]})
            continue
        results.update({
            q: {index: docs[0][1]}
        })

with open("compare_results.yaml", "w") as f:
    yaml.safe_dump(results, f, allow_unicode=True)

   
