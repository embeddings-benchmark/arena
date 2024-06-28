import os
import tqdm
import json
import glob
import argparse

from extractor import _parse_and_clean_wikicode, ENDING_PHRASES


def read_in_cirrus(args, path_to_file: str, output_file: str):
    data = []
    output_f = open(output_file, "w")
    with open(path_to_file, "r") as f:
        # they come in twos
        prev_doc_type = None
        for idx, line in tqdm.tqdm(enumerate(f)):

            if args.debug and idx >= 1000:
                break

            if idx % 2 == 1:
                if prev_doc_type != "_doc":
                    print(f"Skipping non-doc type {prev_doc_type}")
                    continue
                inst = json.loads(line)
                # if inst["title"] == "Scott LeDoux":
                #     breakpoint()
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
    if args.corpus_type == "wikipedia":
        data = read_in_cirrus(args, args.corpus, args.output_dir)
    else:
        raise NotImplementedError("Only wikipedia is supported at the moment")


    # greedily chunk into up to `args.word_limit` words, or at least one paragraph


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