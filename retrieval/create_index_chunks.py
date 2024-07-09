import os
import tqdm
import json
import glob
import argparse

from extractor import _parse_and_clean_wikicode, ENDING_PHRASES


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


def create_chunks(args):
    """
    A wrapper for created a chunked corpus from a given corpus type. Currently only Wikipedia is implemented
        but other corpora can be added.

    :param args: argparse object

    :return: None
    """
    if args.corpus_type == "wikipedia":
        data = read_in_cirrus(args)
    else:
        # TODO add other corpora here
        raise NotImplementedError("Only Wikipedia is supported at the moment")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--corpus_type", type=str, required=True)
    parser.add_argument("-c", "--corpus", type=str, required=True)
    parser.add_argument("-w", "--word_limit", type=int, default=500)
    parser.add_argument("-o", "--output_dir", type=str, default="data/chunks")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    create_chunks(args)

    # python retrieval/create_index_chunks.py -c enwiki-20240624-cirrussearch-content.json -o wiki_extracted.json -t wikipedia