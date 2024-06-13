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
        print("GOT NC: ", ncluster)
        if isinstance(queries, str):
            queries = [queries]

        model = self.load_model(model_name)
        emb = model.encode(queries)

        ### Plotting ###
        import plotly.express as px
        import numpy as np
        import pandas as pd
        from sklearn.decomposition import PCA
        # If only 1 query is given, return a 1D plot
        if len(queries) == 1:
            pca = PCA(n_components=1)
            vis_dims = pca.fit_transform(emb)

            # Put queries & vis_dims into a DataFrame
            df = pd.DataFrame({"text": queries, "x": vis_dims[:, 0]})
            df["text_short"] = df["text"].str[:64]

            hover_data = {
                "text_short": True,
                "x": False,
            }
            if "index" in df.columns:
                hover_data["index"] = False

            fig = px.scatter(
                df, x="x", template="plotly_dark", title="Cluster", hover_data=hover_data
            )
            fig.update_layout(
                xaxis_title='',
                yaxis_title=''
            )

            return fig


        # If 2 or 3 queries are given, return a 2D plot
        # Somehow putting 3 queries in a 3D plot makes the plot very buggy and not look well
        elif len(queries) < 4:
            pca = PCA(n_components=2)
            vis_dims = pca.fit_transform(emb)

            # Put queries & vis_dims into a DataFrame
            df = pd.DataFrame({"text": queries, "x": vis_dims[:, 0], "y": vis_dims[:, 1]})
            df["text_short"] = df["text"].str[:64]

            hover_data = {
                "text_short": True,
                "x": False,
                "y": False,
            }

            fig = px.scatter(
                df, x="x", y="y", template="plotly_dark", title="Cluster", hover_data=hover_data
            )

            return fig

        pca = PCA(n_components=3)
        vis_dims = pca.fit_transform(emb)
        # Put queries & vis_dims into a DataFrame
        import pandas as pd
        data = {"text": queries, "x": vis_dims[:, 0], "y": vis_dims[:, 1], "z": vis_dims[:, 2]}
        hover_data = {
            "text_short": True,
            "x": False,
            "y": False,
            "z": False,
        }

        if ncluster > 1:
            from sklearn.cluster import KMeans
            # Fit KMeans
            cluster_labels = KMeans(n_clusters=ncluster, n_init='auto', random_state=0).fit_predict(emb).tolist()
            hover_data["cluster"] = False
            data["cluster"] = list(map(str,cluster_labels))

        df = pd.DataFrame(data)
        df["text_short"] = df["text"].str[:64]

        if ncluster > 1:
            fig = px.scatter_3d(
                df, x="x", y="y", z="z", color="cluster", template="plotly_dark", hover_data=hover_data,
            )
            # Without legend / colorbar
            fig = fig.update_layout(showlegend=False)
        else:
            fig = px.scatter_3d(
                df, x="x", y="y", z="z", template="plotly_dark", hover_data=hover_data,
            )

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