import concurrent.futures
import os
import random

import mteb
#import spaces

from retrieval.index import build_index, load_or_initialize_index


class ModelManager:
    def __init__(self, model_meta):
        self.model_meta = model_meta["model_meta"]
        self.loaded_models = {}
        self.loaded_indices = {}

    def load_model(self, model_name):
        if model_name in self.loaded_models:
            return self.loaded_models[model_name]
        model = mteb.get_model(model_name)
        self.loaded_models[model_name] = model
        return model

    def load_index(self, model_name, embedbs=32):
        if model_name in self.loaded_indices:
            return self.loaded_indices[model_name]
        meta = self.model_meta.get(model_name, {})

        save_path = "index_" + model_name.replace("/", "_")
        load_index_path = None
        if os.path.exists(save_path):
            load_index_path = save_path

        index, passages = load_or_initialize_index(
            load_index_path=load_index_path,
            dim=meta.get("dim", None),
            limit=meta.get("limit", None),
            index_dtype=meta.get("index_dtype", "bfloat16"),
            passages=["corpus.jsonl"],
        )

        if load_index_path is None:
            build_index(
                self.loaded_models[model_name],
                index,
                passages,
                gpu_embedder_batch_size=embedbs,
            )
            os.makedirs(save_path, exist_ok=True)
            index.save_index(save_path)

        self.loaded_indices[model_name] = index
        return index

    def retrieve_parallel_anon(self, prompt, model_A, model_B):
        if model_A == "" and model_B == "":
            model_names = random.sample(list(self.model_meta.keys()), 2)
        else:
            model_names = [model_A, model_B]

        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [executor.submit(self.retrieve, prompt, model) for model in model_names]
            results = [future.result() for future in futures]
        return results[0], results[1], model_names[0], model_names[1]

    # @spaces.GPU(duration=120)
    def retrieve(self, query, model_name, topk=1):
        model = self.load_model(model_name)
        index = self.load_index(model_name)
        docs, scores = index.search_knn(model.encode([query], convert_to_tensor=True), topk=topk)
        docs = [[query, "Title: " + docs[0][0]["title"] + "\n\n" + "Passage: " + docs[0][0]["text"]]]
        return docs

    def get_model_description_md(self):
        model_description_md = """
| | | |
| ---- | ---- | ---- |
"""
        ct = 0
        for i, name in enumerate(self.model_meta):
            one_model_md = f"[{name}]({self.model_meta[name]['link']}): {self.model_meta[name]['desc']}"
            if ct % 3 == 0:
                model_description_md += "|"
            model_description_md += f" {one_model_md} |"
            if ct % 3 == 2:
                model_description_md += "\n"
            ct += 1
        return model_description_md