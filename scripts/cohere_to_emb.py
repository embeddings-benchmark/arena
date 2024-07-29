import json
from datasets import load_dataset


ds = load_dataset("nreimers/wikipedia_0715_clean_cohere_embed-english-v3.0", split="train")
ds_og = load_dataset("mteb/arena-wikipedia-7-15-24", split="train")

ds = load_dataset("nreimers/stackexchange_cohere_embed-english-v3.0", split="train")
ds_og = load_dataset("mteb/arena-stackexchange", split="train")

assert ds["id"] == ds_og["id"]

passages = ds.rename_columns({"id": "_id"})
with open("emb_stackexchange_embed-english-v3.0.json", 'w') as f:
    embs = []
    for i, line in enumerate(passages):
        if i <= 3811000: continue
        embs += [json.dumps({"id": str(i), "embedding": [str(x) for x in line["emb"]]}) + "\n"]
        if i % 1000 == 0:
            print(i)
            f.writelines(embs)
            embs = []
    f.writelines(embs)
