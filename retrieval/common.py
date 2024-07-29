from datasets import load_dataset


CORPORA = {
    "wikipedia": {"name": "mteb/arena-wikipedia-7-15-24", "columns": {"id": "_id", "text": "text", "title": "title"}},
    "arxiv": {"name": "mteb/arena-arxiv-7-2-24", "columns": {"id": "_id", "abstract": "text", "title": "title"}},
    "stackexchange": {"name": "mteb/arena-stackexchange", "columns": {"id": "_id", "text": "text"}},
}


def load_passages_from_hf(corpus: str, limit: int = None):
    """Returns a list of passages. Each passage is a dict with keys defined in CORPORA"""
    if CORPORA.get(corpus) is None:
        raise NotImplementedError(f"Corpus={corpus} is not found. Try using `wikipedia`, `arxiv`, or `stackexchange`.")
    corpus_dict = CORPORA[corpus]
    ds = load_dataset(corpus_dict['name'], split="train")
    # Rename & remove cols
    ds = ds.rename_columns(corpus_dict['columns'])
    ds = ds.remove_columns([col for col in ds.column_names if col not in corpus_dict['columns'].values()])
    if limit and limit > 1:
        ds = ds.take(limit)
    return ds.to_list()