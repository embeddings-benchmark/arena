from datasets import load_dataset


CORPORA = {
    "wikipedia": {"name": "orionweller/wikipedia-2024-06-24-docs", "columns": {"id": "_id"}},
    "arxiv": {"name": "orionweller/raw_arxiv_7_2_24", "columns": {"id": "_id", "abstract": "text"}},
    "stackexchange": {"name": "orionweller/stackexchange_chunked", "columns": {"id": "_id"}},
}


def load_passages_from_hf(corpus: str, limit: int = None):
    """ 
    Returns a list of passages. Each passage is a dict with the following keys:
    {
        "_id:" doc0,
        "title": "Title 1",
        "text": "Body text 1",
    }
    """
    if CORPORA.get(corpus) is None:
        raise NotImplementedError(f"Corpus={corpus} is not found. Try using `wikipedia`, `arxiv`, or `stackexchange`.")
    corpus_dict = CORPORA[corpus]
    ds = load_dataset(corpus_dict['name'], split="train")
    passages = ds.rename_columns(corpus_dict['columns'])
    if limit and limit > 1:
        passages = passages.take(limit)
    return passages.to_list()