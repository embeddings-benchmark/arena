"""
Remove all votes of a certain model, e.g. if a bug was found.
"""

import json
import os

MODELS_TO_REMOVE = ["text-embedding-004"]
TASKS = ["sts"]

MODELS_TO_REMOVE = ["nomic-embed-text-v1.5"]
TASKS = ["retrieval", "sts", "clustering"]

MODELS_TO_REMOVE = ["Alibaba-NLP/gte-Qwen2-7B-instruct"]
TASKS = ["retrieval", "sts", "clustering"]

for file in os.listdir("results_dataset_to_upload"):
    for task in TASKS:
        if task in file:
            # Load jsonl
            with open(f"results_dataset_to_upload/{file}", "r") as f:
                lines = f.readlines()
            # Remove models
            new_lines = []
            for line in lines:
                line = json.loads(line)
                remove = False
                for model in MODELS_TO_REMOVE:
                    if "model" in line:
                        if model in line["model"]:
                            remove = True
                            break
                    elif "model_name" in line:
                        if model in line["model_name"]:
                            remove = True
                            break
                    elif "0_model_name" in line:
                        if (model in line["0_model_name"]) or (model in line["1_model_name"]):
                            remove = True
                            break
                    else:
                        print(f"Unknown model key in line: {line}")
                if not remove:
                    new_lines.append(line)
            # Save jsonl
            with open(f"results_dataset_to_upload/{file}", "w") as f:
                for line in new_lines:
                    f.write(json.dumps(line) + "\n")

