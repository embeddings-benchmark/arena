import os
import bm25s
import bm25s.hf
from typing import List
import Stemmer
from .common import load_passages_from_hf


from log_utils import build_logger
logger = build_logger("index_logger", "index_logger.log")

class BM25Index:
    def __init__(self, model_name: str, corpus: str = "wikipedia", limit=None):
        self.model_name = model_name
        self.corpus = corpus
        self.repo_name = f"index_{corpus}_{model_name.lower()}"
        self.limit = limit
        self.index = None
        self.stemmer = Stemmer.Stemmer("english")

    def _create_index(self):
        passages = load_passages_from_hf(corpus=self.corpus, limit=self.limit)
        corpus_lst = [r.get("title", "") + " " + r["text"] for r in passages]
        corpus_tokenized = bm25s.tokenize(corpus_lst, stemmer=self.stemmer)
        
        ## By default, bm25s uses method="lucene". See https://github.com/xhluca/bm25s?tab=readme-ov-file#variants.
        retriever = bm25s.hf.BM25HF()
        retriever.index(corpus_tokenized)

        ## Save to hub as a model.
        hf_token = os.getenv("HF_TOKEN")
        retriever.save_to_hub(repo_id=f"mteb/{self.repo_name}", token=hf_token, corpus=passages)
        self.index = retriever
        logger.info(f"Index created and uploaded to `mteb/{self.repo_name}`.")

    def load_index(self):
        """Load the bm25 index or create one if it does not exist."""
        try:
            self.index = bm25s.hf.BM25HF.load_from_hub(
                f"mteb/{self.repo_name}", load_corpus=True, mmap=True
            )
        except:
            logger.warning("Index not found on Huggingface. Creating index.")
            self._create_index()
        logger.info("Index loaded.")

    def search(self, queries: List[str], topk=1):
        """Return topk docs"""
        queries_tokenized = bm25s.tokenize(queries, stemmer=self.stemmer)
        results, scores = self.index.retrieve(queries_tokenized, k=topk)
        return results


if __name__ == "__main__":
    ## To test this, run `python -m retrieval.bm25_index`
    index = BM25Index("BM25", limit=10)
    index.load_index()
    print(index.search(["what is going on?"]))