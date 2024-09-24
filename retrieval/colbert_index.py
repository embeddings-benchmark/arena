"""This is currently under progress pending the RAGatouille overhaul. The script is not meant to be used within the MTEB Arena Space.
However, it is fully functional and can be used as a reference to re-create the ColBERT indexes used in Arena."""

import os
import argparse
from colbert.infra import ColBERTConfig
from colbert import Indexer
import srsly
from .common import load_passages_from_hf

CORPORA = {
    "wikipedia": {
        "name": "mteb/arena-wikipedia-7-15-24",
        "columns": {"id": "_id", "text": "text", "title": "title"},
    },
    "arxiv": {
        "name": "mteb/arena-arxiv-7-2-24",
        "columns": {"id": "_id", "abstract": "text", "title": "title"},
    },
    "stackexchange": {
        "name": "mteb/arena-stackexchange",
        "columns": {"id": "_id", "text": "text"},
    },
}


def make_passage(doc):
    if "title" in doc:
        return doc["title"] + " " + doc["text"]
    return doc["text"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ColBERT indexing script")
    parser.add_argument(
        "--export-mappings",
        action="store_true",
        default=False,
        help="Export document mappings",
    )
    parser.add_argument(
        "--export-dir",
        type=str,
        default="./data/",
        help="Directory to export mappings to",
    )

    args = parser.parse_args()
    data = {}
    for corpus in CORPORA.keys():
        data[corpus] = load_passages_from_hf(corpus)

    models = [
        "/home/azureuser/colbertv2.5_en/FINAL_MODEL_3",
    ]

    for dataset_name, passages in data.items():
        docs = [make_passage(doc) for doc in passages]
        if args.export_mappings:
            int2doc = {}
            int2docid = {}
            for i, doc in enumerate(passages):
                int2doc[i] = docs[i]
                int2docid[i] = doc["_id"]
            srsly.write_json(
                os.path.join(args.export_dir, f"{dataset_name}_int2doc.json"), int2doc
            )
            srsly.write_json(
                os.path.join(args.export_dir, f"{dataset_name}_int2docid.json"),
                int2docid,
            )
        print(f"Indexing {len(docs)} documents for {dataset_name}", flush=True)

        config = ColBERTConfig(
            nbits=2,
            nranks=1,
            root=".experiments/",
            avoid_fork_if_possible=True,
            overwrite=True,
            kmeans_niters=10,
            bsize=512,
            index_bsize=512,
            doc_maxlen=512,
        )
        indexer = Indexer(
            checkpoint="answerdotai/answerai-colbert-small-v1", config=config
        )
        indexer.index(
            name=f"{dataset_name}_V1_AAIColBERT-small", collection=docs, overwrite=True
        )
