import os
import tqdm
import json
import glob
import argparse
from datasets import load_dataset
import gzip

from extractor import _parse_and_clean_wikicode, ENDING_PHRASES
from newest_arxiv import create_newest_arxiv

def read_in_cirrus(args):
    """
    Reads in a Cirrus dump from Wikipedia (https://dumps.wikimedia.org/other/cirrussearch/) and extracts the text
        from the pages. The text is then chunked into paragraphs of a certain length and written out in JSONL format.

    :param args: argparse object (containing the path to the file, the output file, and the word limit)

    :return: None
    """
    data = []
    output_f = open(args.output_file, "w")
    with open(args.path_to_file, "r") as f:
        # they come in twos
        prev_doc_type = None
        for idx, line in tqdm.tqdm(enumerate(f)):

            if args.debug and idx >= 5000:
                break

            if idx % 2 == 1:
                if prev_doc_type != "_doc":
                    print(f"Skipping non-doc type {prev_doc_type}")
                    continue
                
                inst = json.loads(line)
                text = _parse_and_clean_wikicode(inst["source_text"])


                final_paragraphs = []
                cur_words = 0
                cur_str = ""
                # greedily bring them together from a backwards approach, so that heading fit with their paragraphs
                for paragraph in reversed(text.split("\n")):
                    if paragraph.strip() in [""] + ENDING_PHRASES or paragraph.startswith("Category:"):
                        continue

                    words = paragraph.split()
                    # if it's a title, we want to try and include it
                    is_short_and_barely_over = (cur_words + len(words) > args.word_limit) and len(words) < 5 and (cur_words < args.word_limit + 10)
                    if (cur_words + len(words) > args.word_limit) and not is_short_and_barely_over:
                        final_paragraphs.insert(0, cur_str.strip())
                        cur_str = ""
                        cur_words = 0
                    else:
                        cur_str = paragraph + "\n" + cur_str
                        cur_words += len(words)

                if cur_str.strip() != "":
                    final_paragraphs.insert(0, cur_str.strip())


                for idx, paragraph in enumerate(final_paragraphs):
                    processed_data = {
                        "title": inst["title"],
                        "id": str(inst["page_id"]) + "-" + str(idx),
                        "text": paragraph,
                    }
                    output_f.write(json.dumps(processed_data) + "\n")

            else:
                prev_doc_type = json.loads(line)["index"]["_type"]


def prep_stackexchange(args):
    """
    Creates the index for stackexchange, limiting the corpus to instances less than `args.word_limit`.
        The `args.corpus` directory should be a folder full of files from the original Dolma setup
        See https://huggingface.co/datasets/orionweller/stackexchange_redpajamas for an example

    NOTE: this corpus is large and we only filter about 700k out of ~22 million with a word limit of 500
    """
    print(f"Loading stackexchange dataset")
    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir)
    output_f = open(os.path.join(args.output_dir, "train.jsonl"), "w")

    written_out = 0
    skipped = 0
    data = []
    for file in tqdm.tqdm(glob.glob(os.path.join(args.corpus, "*.json.gz"))):
        with gzip.open(file, "rb") as f:
            for line in tqdm.tqdm(f):
                inst = json.loads(line)
                if len(inst["text"].split(" ")) > args.word_limit:
                    skipped += 1
                    continue
                output_f.write(json.dumps({
                    "text": inst["text"],
                    "id": inst["id"],
                }) + "\n")
                written_out += 1

    print(f"Skipped {skipped} and wrote out {written_out}")


def create_chunks(args):
    """
    A wrapper for created a chunked corpus from a given corpus type. Currently only Wikipedia is implemented
        but other corpora can be added.

    :param args: argparse object

    :return: None
    """
    if args.corpus_type == "wikipedia":
        read_in_cirrus(args)
    elif args.corpus_type == "arxiv":
        # make sure the file is downloaded (e.g. "arxiv-metadata-oai-snapshot.json") see `newest_arxiv.py` for details
        assert os.path.isfile(args.corpus), f"File {args.corpus} does not exist"
        create_newest_arxiv(args)
    elif args.corpus_type == "stackexchange":
        # make sure that the repo exists and has *.json.gz files
        # NOTE: cannot stream from HF due to issues with metadata in the original not being consistent (pyarrow)
        assert os.path.isdir(args.corpus), f"Directory {args.corpus} does not exist"
        prep_stackexchange(args)
    else:
        raise NotImplementedError(f"Corpus type {args.corpus_type} not implemented")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--corpus_type", type=str, required=True)
    parser.add_argument("-c", "--corpus", type=str, required=True)
    parser.add_argument("-w", "--word_limit", type=int, default=500)
    parser.add_argument("-o", "--output_dir", type=str, default="data/chunks")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    create_chunks(args)

    # example usages:
    #   python retrieval/create_index_chunks.py -c enwiki-20240624-cirrussearch-content.json -o wiki_extracted.json -t wikipedia
    #   python retrieval/create_index_chunks.py -c stackexchange_redpajamas -o stackexchange_extracted -t stackexchange
    #   python retrieval/create_index_chunks.py -c arxiv-metadata-oai-snapshot.json -o raw_arxiv_newlines -t arxiv