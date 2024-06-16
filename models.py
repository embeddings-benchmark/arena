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
        self.loaded_samples = {}

    def load_model(self, model_name):
        if model_name in self.loaded_models:
            return self.loaded_models[model_name]
        model = mteb.get_model(model_name, revision=self.model_meta[model_name].get("revision", None))
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
    
    def retrieve_draw(self):
        if "retrieval" not in self.loaded_samples:
            from datasets import load_dataset
            self.loaded_samples["retrieval"] = load_dataset("mteb/nq", "queries")["queries"]["text"]
        return random.choice(self.loaded_samples["retrieval"])

    def clustering_draw(self):
        if "clustering" not in self.loaded_samples:
            from datasets import load_dataset
            ds = load_dataset("mteb/reddit-clustering", split="test")
            #ds = load_dataset("mteb/twentynewsgroups-clustering", split="test")
            self.loaded_samples["clustering"] = []
            for s, l in zip(ds["sentences"], ds["labels"]):
                # Limit to 8 labels to avoid having every sample stem from a different cluster
                rand_clusters = random.sample(l, 4)
                self.loaded_samples["clustering"].append([(x, y) for x, y in zip(s, l) if y in rand_clusters])
        samples = random.sample(random.choice(self.loaded_samples["clustering"]), random.randint(4, 16))
        return "<|SEP|>".join([x for x, y in samples]), len(set(y for x, y in samples))

    def sts_draw(self):
        if "sts" not in self.loaded_samples:
            from datasets import load_dataset
            self.loaded_samples["sts"] = load_dataset("sentence-transformers/all-nli", "triplet", split="test")
        samples = list(random.choice(self.loaded_samples["sts"]).values())
        random.shuffle(samples) # Randomly permute order of the three samples
        return samples


    def retrieve_parallel(self, prompt, model_A, model_B):
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
    
    def clustering_parallel(self, prompt, model_A, model_B, ncluster=1):
        if model_A == "" and model_B == "":
            model_names = random.sample(list(self.model_meta.keys()), 2)
        else:
            model_names = [model_A, model_B]

        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [executor.submit(self.clustering, prompt, model, ncluster) for model in model_names]
            results = [future.result() for future in futures]
        return results[0], results[1], model_names[0], model_names[1]
    
    def clustering(self, queries, model_name, ncluster=1, method="PCA"):
        """
        Sources:
        - https://www.gradio.app/guides/plot-component-for-maps
        - https://github.com/openai/openai-cookbook/blob/main/examples/Visualizing_embeddings_in_3D.ipynb
        - https://www.nlplanet.org/course-practical-nlp/02-practical-nlp-first-tasks/12-clustering-articles
        """
        import pandas as pd
        import plotly.express as px
        from sklearn.cluster import KMeans
        from sklearn.decomposition import PCA

        if len(queries) == 1:
            # No need to do PCA; just return a 1D plot
            df = pd.DataFrame({"txt": queries, "x": [0]})
            df["txt"] = df["txt"].str[:90]
            fig = px.scatter(df, x="x", template="plotly_dark", hover_name="txt")
            fig.update_layout(xaxis_title='', yaxis_title='')
        elif len(queries) < 4:
            model = self.load_model(model_name)
            emb = model.encode(queries)
            vis_dims = PCA(n_components=2).fit_transform(emb)
            df = pd.DataFrame({"txt": queries, "x": vis_dims[:, 0], "y": vis_dims[:, 1]})
            df["txt"] = df["txt"].str[:90]
            fig = px.scatter(df, x="x", y="y", template="plotly_dark", hover_name="txt")
        else:
            model = self.load_model(model_name)
            emb = model.encode(queries)
            vis_dims = PCA(n_components=3).fit_transform(emb)
            data = {"txt": queries, "x": vis_dims[:, 0], "y": vis_dims[:, 1], "z": vis_dims[:, 2]}
            if ncluster > 1:
                data["cluster"] = KMeans(n_clusters=ncluster, n_init='auto', random_state=0).fit_predict(emb).tolist()
            df = pd.DataFrame(data)
            df["txt"] = df["txt"].str[:90]
            if ncluster > 1:
                fig = px.scatter_3d(df, x="x", y="y", z="z", color="cluster", template="plotly_dark", hover_name="txt")
            else:
                fig = px.scatter_3d(df, x="x", y="y", z="z", template="plotly_dark", hover_name="txt")
        fig.update_traces(
            hovertemplate="<b>%{hovertext}</b><extra></extra>",
            hovertext=df["txt"].tolist()
        )
        fig.update_layout(showlegend=False, coloraxis_showscale=False) # Remove legend / colorbar
        return fig

    def sts_parallel(self, txt0, txt1, txt2, model_A, model_B):
        if model_A == "" and model_B == "":
            model_names = random.sample(list(self.model_meta.keys()), 2)
        else:
            model_names = [model_A, model_B]

        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [executor.submit(self.sts, txt0, txt1, txt2, model) for model in model_names]
            results = [future.result() for future in futures]
        return results[0], results[1], model_names[0], model_names[1]
        
    def sts(self, txt0, txt1, txt2, model_name):
        import numpy as np
        from numpy.linalg import norm

        model = self.load_model(model_name)
        # Compute cos sim all texts; Shape: [3, dim]
        emb0, emb1, emb2 = model.encode([txt0, txt1, txt2])

        cos_sim_01 = (emb0 @ emb1.T) / (norm(emb0)*norm(emb1))
        cos_sim_02 = (emb0 @ emb2.T) / (norm(emb0)*norm(emb2))
        cos_sim_12 = (emb1 @ emb2.T) / (norm(emb1)*norm(emb2))

        print(f"cos_sim_01: {cos_sim_01}, cos_sim_02: {cos_sim_02}, cos_sim_12: {cos_sim_12}")

        # Normalize the cosine similarities so that they sum to 1
        # cos_sims = np.array([cos_sim_01, cos_sim_02, cos_sim_12])
        # cos_sims /= cos_sims.sum()
        # Normalize the cosine similarities into a range from 0.5 to 1
        # cos_sims = np.array([cos_sim_01, cos_sim_02, cos_sim_12])
        # cos_sims = (cos_sims - cos_sims.min()) / (cos_sims.max() - cos_sims.min()) * 0.5 + 0.5
        # Normalize the cosine similarities into a range from 1 to 0.5 (reverse of above such that higher similarity means lower)
        cos_sims = np.array([cos_sim_01, cos_sim_02, cos_sim_12])
        cos_sims = 1 - (cos_sims - cos_sims.min()) / (cos_sims.max() - cos_sims.min()) * 0.5

        print(f"cos_sims: {cos_sims}")

        # Scale up the normalized values for better visualization
        cos_sims *= 200

        # Calculate positions of the points in 2D space
        A = (0, 0)
        B = (cos_sims[0], 0)

        # Use the law of cosines to find the third point
        a = cos_sims[0]
        b = cos_sims[1]
        c = cos_sims[2]

        # Calculate coordinates of point C using the law of cosines
        cos_angle = (a**2 + b**2 - c**2) / (2 * a * b)
        angle = np.arccos(cos_angle)

        Cx = b * np.cos(angle)
        Cy = b * np.sin(angle)
        C = (Cx, Cy)

        print(f"A: {A}, B: {B}, C: {C}")

        # Create HTML string with SVG to display the triangle
        html_string = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Cosine Similarity Triangle</title>
</head>
<body>
    <svg width="400" height="400" viewBox="-150 -50 400 400">
        <!-- Draw the triangle -->
        <line x1="{A[0]}" y1="{A[1]}" x2="{B[0]}" y2="{B[1]}" stroke="black" stroke-width="2"/>
        <line x1="{B[0]}" y1="{B[1]}" x2="{C[0]}" y2="{C[1]}" stroke="black" stroke-width="2"/>
        <line x1="{C[0]}" y1="{C[1]}" x2="{A[0]}" y2="{A[1]}" stroke="black" stroke-width="2"/>
        
        <!-- Draw the points -->
        <circle cx="{A[0]}" cy="{A[1]}" r="8" fill="red"/>
        <circle cx="{B[0]}" cy="{B[1]}" r="8" fill="green"/>
        <circle cx="{C[0]}" cy="{C[1]}" r="8" fill="blue"/>
        
        <!-- Label the points -->
        <text x="{A[0] + 5}" y="{A[1] - 5}" font-family="Arial" font-size="40" fill="black">(1)</text>
        <text x="{B[0] + 5}" y="{B[1] - 5}" font-family="Arial" font-size="40" fill="black">(2)</text>
        <text x="{C[0] + 5}" y="{C[1] + 20}" font-family="Arial" font-size="40" fill="black">(3)</text>
    </svg>
</body>
</html>
"""
        return html_string

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