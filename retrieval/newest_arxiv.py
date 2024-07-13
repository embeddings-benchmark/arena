"""Take data from https://www.kaggle.com/datasets/Cornell-University/arxiv?resource=download and
only keep useful information
Adapted from https://github.com/embeddings-benchmark/mteb/blob/main/scripts/data/arxiv/script_raw.py
"""

from __future__ import annotations

import gzip
import json

import jsonlines
from tqdm import tqdm
import os
import re


def clean_text_newlines(text):
    # First, replace double newlines with a special marker
    text = re.sub(r'\n\s*\n', '<<PARAGRAPH>>', text)
    
    # Replace all other whitespace (including single newlines) with a single space
    text = re.sub(r'\s+', ' ', text)
    
    # Restore double newlines
    text = text.replace('<<PARAGRAPH>>', '\n\n').strip()
    return text



def create_newest_arxiv(args):
    """
    Takes a dump from the newest arxiv and turns it into a format that can be used for indexing.
        Download it via https://www.kaggle.com/datasets/Cornell-University/arxiv and unzip
        Then run this script. Upload the .json.gz files to a Hugginface Dataset
    """
    os.makedirs(args.output_dir, exist_ok=True)

    with open(args.corpus, "r") as file:
        old_lines = file.readlines()
        new_lines = []
        split = 0

        for idx, line in enumerate(tqdm(old_lines)):
            # Write split each 100k lines
            if idx > 0 and idx % 100000 == 0:
                file_name = f"{args.output_dir}/train_{split}"
                with jsonlines.open(f"{file_name}.jsonl", "w") as writer:
                    writer.write_all(new_lines)
                with open(f"{file_name}.jsonl", "rb") as f_in:
                    with gzip.open(f"{file_name}.jsonl.gz", "wb") as f_out:
                        f_out.writelines(f_in)
                new_lines = []
                split += 1

            old_json = json.loads(line)
            new_json = {
                "id": old_json["id"],
                "title": old_json["title"],
                "abstract": clean_text_newlines(old_json["abstract"]),
                "categories": old_json["categories"],
            }
            new_lines.append(new_json)

        file_name = f"{args.output_dir}/train_{split}"
        with jsonlines.open(f"{file_name}.jsonl", "w") as writer:
            writer.write_all(new_lines)
        with open(f"{file_name}.jsonl", "rb") as f_in:
            with gzip.open(f"{file_name}.jsonl.gz", "wb") as f_out:
                f_out.writelines(f_in)