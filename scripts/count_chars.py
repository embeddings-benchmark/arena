from datasets import load_dataset

CORPORA = {
    "wikipedia": {"name": "orionweller/wikipedia-2024-06-24-docs", "columns": {"id": "_id"}},
    "arxiv": {"name": "orionweller/arxiv_7_2_24", "columns": {"id": "_id", "abstract": "text"}},
    "stackexchange": {"name": "orionweller/stackexchange_chunked", "columns": {"id": "_id"}},
}


ds = load_dataset(CORPORA["wikipedia"]["name"], split="train")
# Count characters
total_chars = 0 # 4538345526
for i in range(len(ds)):
    total_chars += len(ds[i]["title"] + " " + ds[i]["text"])
print(total_chars)

ds = load_dataset(CORPORA["arxiv"]["name"], split="train")
# Count characters
total_chars = 0 # 2579266272
for i in range(len(ds)):
    total_chars += len(ds[i]["title"] + " " + ds[i]["abstract"])
print(total_chars)
