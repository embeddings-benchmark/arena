import concurrent.futures
import os
import math
import random
import threading

import mteb
import spaces
import torch

from log_utils import build_logger
from retrieval.index import build_index, load_or_initialize_index
from retrieval.index import DistributedIndex
from retrieval.gcp_index import VertexIndex
from retrieval.bm25_index import BM25Index
from clustering_samples import CLUSTERING_CATEGORIES

logger = build_logger("model_logger", "model_logger.log")

MODEL_TO_CUDA_DEVICE = {
    "sentence-transformers/all-MiniLM-L6-v2": "0",
    "nomic-ai/nomic-embed-text-v1.5": "0",
    "intfloat/multilingual-e5-large-instruct": "1",
    "intfloat/e5-mistral-7b-instruct": "2",
    "GritLM/GritLM-7B": "3",
    "BAAI/bge-large-en-v1.5": "4",
    "Alibaba-NLP/gte-Qwen2-7B-instruct": "5",
    "Salesforce/SFR-Embedding-2_R": "6",
    "jinaai/jina-embeddings-v2-base-en": "7",
    "mixedbread-ai/mxbai-embed-large-v1": "7",
}

CORPUS_TO_FORMAT = {
    "arxiv": "Title: {title}\n\nAbstract: {text}",
#    "wikipedia": "Title: {title}\n\nPassage: {text}",
    "wikipedia": "{title}\n\n{text}",
    "stackexchange": "{text}",
}

class ModelManager:
    def __init__(self, model_meta, use_gcp_index: bool = False, load_all: bool = False):
        self.model_meta = model_meta["model_meta"]
        self.models_retrieval = sorted(set(model_meta["model_meta"].keys()))
        self.models_retrieval_stackexchange = [
            "sentence-transformers/all-MiniLM-L6-v2",
            "nomic-ai/nomic-embed-text-v1.5",
            "mixedbread-ai/mxbai-embed-large-v1",
            "jinaai/jina-embeddings-v2-base-en",
            "Salesforce/SFR-Embedding-2_R",
            "GritLM/GritLM-7B",
            "BAAI/bge-large-en-v1.5",
            "intfloat/multilingual-e5-large-instruct",
            "intfloat/e5-mistral-7b-instruct",
            "voyage-multilingual-2",
            "BM25",
        ]
        self.models_sts = sorted(set(model_meta["model_meta"].keys()) - set(["BM25"]))
        self.models_clustering = sorted(set(model_meta["model_meta"].keys()) - set(["BM25"]))
        self.use_gcp_index = use_gcp_index
        self.loaded_models = {}
        self.loaded_indices = {}
        self.loaded_samples = {}
        self.lock = threading.Lock()
        if load_all:
            for model_name in self.models_sts:
                self.load_model(model_name)
            # Load BM25 indices
            self.load_bm25_index("BM25", "wikipedia")
            self.load_bm25_index("BM25", "arxiv")
            self.load_bm25_index("BM25", "stackexchange")
            # Load GCP indices
            if use_gcp_index:
                for model_name in self.models_retrieval:
                    if model_name == "BM25": continue
                    self.load_gcp_index(model_name, "wikipedia")
                    self.load_gcp_index(model_name, "arxiv")
                for model_name in self.models_retrieval_stackexchange:
                    if model_name == "BM25": continue
                    self.load_gcp_index(model_name, "stackexchange")
            # Load random samples
            self.retrieve_draw()
            self.clustering_draw()
            self.sts_draw()

    def load_model(self, model_name):
        if model_name in self.loaded_models:
            return self.loaded_models[model_name]
        # Do not allow this function to be run by processes in parallel but always one by one
        # which is needed due to a bug in transformers where it temporarily sets a default torch dtype
        # so if two models are loaded in parallel & have different dtypes, one will have the wrong dtype
        with self.lock:
            logger.info(f"Loading & caching model: {model_name}")
            device = "cpu"
            if torch.cuda.is_available():
                device = "cuda"
                if model_name in MODEL_TO_CUDA_DEVICE:
                    device += ":" + MODEL_TO_CUDA_DEVICE[model_name]
            model = mteb.get_model(
                model_name,
                revision=self.model_meta[model_name].get("revision", None),
                device=device,
            )
            self.loaded_models[model_name] = model
            return model

    def load_local_index(self, model_name, corpus, embedbs=32) -> DistributedIndex:
        """Load local index into memory. Create index if it does not exist."""
        if (model_name in self.loaded_indices) and (corpus in self.loaded_indices[model_name]):
            return self.loaded_indices[model_name][corpus]
        meta = self.model_meta.get(model_name, {})

        save_path = "index_" + corpus + "_" + model_name.replace("/", "_")
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

        self.loaded_indices.setdefault(model_name, {})
        self.loaded_indices[model_name][corpus] = index
        return index
    
    def load_gcp_index(self, model_name, corpus) -> VertexIndex:
        if (model_name in self.loaded_indices) and (corpus in self.loaded_indices[model_name]):
            return self.loaded_indices[model_name][corpus]
        meta = self.model_meta.get(model_name, {})
        index = VertexIndex(
            dim=meta.get("dim", None),
            model_name=model_name,
            model=self.loaded_models[model_name],
            corpus=corpus,
            limit=meta.get("limit", None)
        )
        index.load_endpoint()
        self.loaded_indices.setdefault(model_name, {})
        self.loaded_indices[model_name][corpus] = index
        return index

    def load_bm25_index(self, model_name:str, corpus:str, limit=None) -> BM25Index:
        if model_name in self.loaded_indices:
            if corpus in self.loaded_indices[model_name]:
                return self.loaded_indices[model_name][corpus]
        index = BM25Index(
            model_name=model_name, 
            corpus=corpus, 
            limit=limit
        )
        index.load_index()
        self.loaded_indices.setdefault(model_name, {})
        self.loaded_indices[model_name][corpus] = index
        return index

    def retrieve_draw(self, corpora=["wikipedia", "arxiv", "stackexchange"]):
        if "retrieval" not in self.loaded_samples:
            self.loaded_samples["retrieval"] = {}
            from datasets import load_dataset
            self.loaded_samples["retrieval"]["wikipedia"] = load_dataset("mteb/nq", "queries", split="queries")["text"]
            self.loaded_samples["retrieval"]["arxiv"] = load_dataset("mteb/arena-arxiv-7-2-24-samples", split="train")["query"]
            self.loaded_samples["retrieval"]["stackexchange"] = [x["query"] for sub_ds in load_dataset("colbertv2/lotte", "pooled", split=['search_dev', 'search_test']) for x in sub_ds]
        corpus = random.choice(corpora)
        return random.choice(self.loaded_samples["retrieval"][corpus]), corpus

    def clustering_draw(self):
        if "clustering" not in self.loaded_samples:
            self.loaded_samples["clustering"] = []
            for i in range(10000):
                # Select 2-5 categories 
                n_categories = random.randint(2, 5)
                sampled_categories = random.sample(list(CLUSTERING_CATEGORIES.keys()), n_categories)
                sampled_items = []
                for category in sampled_categories:
                    # Add all items from the selected category
                    #sampled_items.extend(random.sample(CLUSTERING_CATEGORIES[category], random.randomint(2, min(CLUSTERING_CATEGORIES[category], 8))))
                    # randomint does not exist; fix
                    sampled_items.extend(random.sample(CLUSTERING_CATEGORIES[category], random.randint(2, min(len(CLUSTERING_CATEGORIES[category]), 8))))
                
                self.loaded_samples["clustering"].append((sampled_items, n_categories))
        
        # Randomly select one of the pre-generated samples
        selected_sample = random.choice(self.loaded_samples["clustering"])
        return "<|SEP|>".join(selected_sample[0]), selected_sample[1]

    def sts_draw(self):
        if "sts" not in self.loaded_samples:
            from datasets import load_dataset
            self.loaded_samples["sts"] = load_dataset("sentence-transformers/all-nli", "triplet", split="test")
        samples = list(random.choice(self.loaded_samples["sts"]).values())
        random.shuffle(samples) # Randomly permute order of the three samples
        return samples

    def retrieve_parallel(self, prompt, corpus, model_A, model_B):
        if model_A == "" and model_B == "":
            if corpus == "stackexchange":
                model_names = random.sample(self.models_retrieval_stackexchange, 2)
            else:
                model_names = random.sample(self.models_retrieval, 2)
        else:
            model_names = [model_A, model_B]

        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [executor.submit(self.retrieve, prompt, corpus, model) for model in model_names]
            results = [future.result() for future in futures]
        return results[0], results[1], model_names[0], model_names[1]

    @spaces.GPU(duration=120)
    def retrieve(self, query, corpus, model_name, topk=1):
        corpus_format = CORPUS_TO_FORMAT[corpus]

        if "BM25" in model_name:
            index = self.load_bm25_index("BM25", corpus)
            docs = index.search([query], topk=topk)
            if corpus == "stackexchange":
                return [[query, corpus_format.format(text=docs[0][0]["text"])]]
            else:
                return [[query, corpus_format.format(title=docs[0][0]["title"], text=docs[0][0]["text"])]]
            
        model = self.load_model(model_name)
        kwargs = {} if self.use_gcp_index else {"convert_to_tensor": True}
        if f"instruction_query_{corpus}" in self.model_meta[model_name]:
            kwargs["instruction"] = self.model_meta[model_name][f"instruction_query_{corpus}"]
            logger.info(f"Using instruction: {kwargs['instruction']}")
        # Optionally time embedding & search
        # import time
        # x = time.time()
        if hasattr(model, "encode_queries"):
            query_embed = model.encode_queries([query], **kwargs)
        else:
            query_embed = model.encode([query], **kwargs)

        if self.use_gcp_index:
            # y = time.time()
            # logger.info(f"Embedding time: {y - x}")
            index = self.load_gcp_index(model_name, corpus)
            # z = time.time()
            # logger.info(f"Loading time: {z - y}")
            docs = index.search(query_embeds=query_embed.tolist(), topk=topk)
            # logger.info(f"Search time: {time.time() - z}")
            docs = [[query, corpus_format.format(title=docs[0].get("title", ""), text=docs[0]["text"])]]
        else:
            index = self.load_local_index(model_name, corpus)
            docs, scores = index.search_knn(query_embed, topk=topk)
            docs = [[query, corpus_format.format(title=docs[0].get("title", ""), text=docs[0][0]["text"])]]
        return docs
    
    def clustering_parallel(self, prompt, model_A, model_B, ncluster=1, ndim="3D", dim_method="PCA", clustering_method="KMeans"):
        if model_A == "" and model_B == "":
            model_names = random.sample(self.models_clustering, 2)
        else:
            model_names = [model_A, model_B]

        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [executor.submit(self.clustering, prompt, model, ncluster, ndim, dim_method, clustering_method, False) for model in model_names]
            results = [future.result() for future in futures]
        return results[0], results[1], model_names[0], model_names[1]

    @spaces.GPU(duration=120)
    def clustering(self, queries, model_name, ncluster=1, ndim="3D", dim_method="PCA", clustering_method="KMeans", single_ui=True):
        """
        Sources:
        - https://www.gradio.app/guides/plot-component-for-maps
        - https://github.com/openai/openai-cookbook/blob/main/examples/Visualizing_embeddings_in_3D.ipynb
        - https://www.nlplanet.org/course-practical-nlp/02-practical-nlp-first-tasks/12-clustering-articles
        """
        import pandas as pd
        import plotly.express as px
        from sklearn.cluster import KMeans, MiniBatchKMeans
        from sklearn.decomposition import PCA
        from sklearn.manifold import TSNE
        from umap import UMAP

        model_kwargs = {} if self.use_gcp_index else {"convert_to_tensor": True}
        if model_name == "text-embedding-004":
            model_kwargs["google_task_type"] = "CLUSTERING"
        elif model_name == "embed-english-v3.0":
            model_kwargs["cohere_task_type"] = "clustering"
        elif model_name in ["nomic-ai/nomic-embed-text-v1.5", "nomic-ai/nomic-embed-text-v1"]:
            model_kwargs["input_type"] = "clustering"

        cutoff = 178 if single_ui else 88

        if len(queries) == 1:
            # No need to do PCA; just return a 1D plot
            df = pd.DataFrame({"txt": queries, "x": [0]})
            df["txt"] = df["txt"].str[:cutoff]
            fig = px.scatter(df, x="x", template="plotly_dark", hover_name="txt")
            fig.update_layout(xaxis_title='', yaxis_title='')
        elif (ndim == "2D") or (len(queries) < 4):
            model = self.load_model(model_name)
            emb = model.encode(queries, **model_kwargs)
            if dim_method == "UMAP":
                vis_dims = UMAP(n_components=2).fit_transform(emb)
            elif dim_method == "TSNE":
                vis_dims = TSNE(n_components=2, perplexity=min(30.0, len(queries)//2)).fit_transform(emb)
            else:
                vis_dims = PCA(n_components=2).fit_transform(emb)
            data = {"txt": queries, "x": vis_dims[:, 0], "y": vis_dims[:, 1]}
            if ncluster > 1:
                if clustering_method == "MiniBatchKMeans":
                    data["cluster"] = MiniBatchKMeans(n_clusters=ncluster, n_init="auto", random_state=0).fit_predict(emb).tolist()
                else:
                    data["cluster"] = KMeans(n_clusters=ncluster, n_init='auto', random_state=0).fit_predict(emb).tolist()
            df = pd.DataFrame(data)
            df["txt"] = df["txt"].str[:cutoff]
            if ncluster > 1:
                fig = px.scatter(df, x="x", y="y", color="cluster", template="plotly_dark", hover_name="txt")
            else:
                fig = px.scatter(df, x="x", y="y", template="plotly_dark", hover_name="txt")
            fig.update_traces(marker=dict(size=12))
        else:
            model = self.load_model(model_name)
            emb = model.encode(queries, **model_kwargs)
            if dim_method == "UMAP":
                vis_dims = UMAP(n_components=3).fit_transform(emb)
            elif dim_method == "TSNE":
                vis_dims = TSNE(n_components=3, perplexity=min(30.0, len(queries)//2)).fit_transform(emb)
            else:
                vis_dims = PCA(n_components=3).fit_transform(emb)
            data = {"txt": queries, "x": vis_dims[:, 0], "y": vis_dims[:, 1], "z": vis_dims[:, 2]}
            if ncluster > 1:
                if clustering_method == "MiniBatchKMeans":
                    data["cluster"] = MiniBatchKMeans(n_clusters=ncluster, n_init="auto", random_state=0).fit_predict(emb).tolist()
                else:
                    data["cluster"] = KMeans(n_clusters=ncluster, n_init='auto', random_state=0).fit_predict(emb).tolist()
            df = pd.DataFrame(data)
            df["txt"] = df["txt"].str[:cutoff]
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
            model_names = random.sample(self.models_sts, 2)
        else:
            model_names = [model_A, model_B]

        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [executor.submit(self.sts, txt0, txt1, txt2, model) for model in model_names]
            results = [future.result() for future in futures]
        return results[0], results[1], model_names[0], model_names[1]

    @spaces.GPU(duration=120)
    def sts(self, txt0, txt1, txt2, model_name):
        import numpy as np
        from numpy.linalg import norm
        import plotly.graph_objects as go

        model_kwargs = {} if self.use_gcp_index else {"convert_to_tensor": True}
        if model_name == "text-embedding-004":
            model_kwargs["google_task_type"] = "SEMANTIC_SIMILARITY"
        elif model_name in ["nomic-ai/nomic-embed-text-v1.5", "nomic-ai/nomic-embed-text-v1"]:
            model_kwargs["input_type"] = "classification"
        # Cohere has no specific task type for STS
        # elif model_name == "embed-english-v3.0":
        #     model_kwargs["cohere_task_type"] =
    
        model = self.load_model(model_name)
        # Compute cos sim all texts; Shape: [3, dim]
        emb0, emb1, emb2 = model.encode([txt0, txt1, txt2], **model_kwargs)

        cos_sim_01 = (emb0 @ emb1.T) / (norm(emb0)*norm(emb1))
        cos_sim_02 = (emb0 @ emb2.T) / (norm(emb0)*norm(emb2))
        cos_sim_12 = (emb1 @ emb2.T) / (norm(emb1)*norm(emb2))

        # Normalize the cosine similarities so that they sum to 1
        # cos_sims = np.array([cos_sim_01, cos_sim_02, cos_sim_12])
        # cos_sims /= cos_sims.sum()
        # Normalize the cosine similarities into a range from 0.5 to 1
        # cos_sims = np.array([cos_sim_01, cos_sim_02, cos_sim_12])
        # cos_sims = (cos_sims - cos_sims.min()) / (cos_sims.max() - cos_sims.min()) * 0.5 + 0.5
        # Normalize the cosine similarities into a range from 1 to 0.5 (reverse of above such that higher similarity means lower)
        cos_sims = np.array([cos_sim_01, cos_sim_02, cos_sim_12])
        cos_sims = 1 - (cos_sims - cos_sims.min()) / (cos_sims.max() - cos_sims.min()) * 0.5

        # Scale up the normalized values for better visualization
        cos_sims *= 200

        # Calculate positions of the points in 2D space
        A = (0, 0)
        B = (cos_sims[0], 0)

        # Given distances
        c = cos_sims[0]
        b = cos_sims[1]
        a = cos_sims[2]

        # Compute angle at A
        # https://en.wikipedia.org/wiki/Law_of_cosines#Use_in_solving_triangles
        angle_A = math.acos((b**2 + c**2 - a**2) / (2 * b * c))
        # https://en.wikipedia.org/wiki/Law_of_cosines#Cartesian_coordinates
        C_x = b * math.cos(angle_A)
        C_y = b * math.sin(angle_A)

        C = (C_x, C_y)

        # Create Plotly plot
        fig = go.Figure()

        # Add lines for the triangle
        fig.add_trace(go.Scatter(
            x=[A[0], B[0], C[0], A[0]],
            y=[A[1], B[1], C[1], A[1]],
            mode='lines',
            line=dict(color='black', width=2),
            showlegend=False,
            hoverinfo='none',  # Disable hoverinfo for lines
        ))        

        # Add points for the vertices with hover information
        fig.add_trace(go.Scatter(
            x=[A[0], B[0], C[0]],
            y=[A[1], B[1], C[1]],
            mode='markers+text',
            text=['(1)', '(2)', '(3)'],
            textposition='top center',
            hovertext=[txt0, txt1, txt2],
            hoverinfo='text',
            textfont=dict(size=16),
            marker=dict(size=20, color=['#f6511d', '#ffb400', '#00a6ed']),
            showlegend=False
        ))
        # Calculate distances for annotation
        distances = [
            f"{round(c)}",
            f"{round(b)}",
            f"{round(a)}"
        ] 
        # Calculate midpoints for annotation placement
        midpoints = [
            ((A[0] + B[0]) / 2, (A[1] + B[1]) / 2),
            ((A[0] + C[0]) / 2, (A[1] + C[1]) / 2),
            ((B[0] + C[0]) / 2, (B[1] + C[1]) / 2),
        ]        
        # Add distance annotations
        for i, (x, y) in enumerate(midpoints):
            fig.add_trace(go.Scatter(
                x=[x],
                y=[y],
                mode='text',
                text=[distances[i]],
                textposition='top center',
                textfont=dict(size=20, color='black', family='Arial', weight='bold'),
                showlegend=False,
                hoverinfo='none'
            ))

        # Update layout
        fig.update_layout(
            # Do not put title so there is more space for the plot; does not seem to add value anyways
            # title='Similarity Triangle',
            xaxis=dict(
                visible=False,
                scaleanchor='y',  # Anchor x-axis scale to y-axis
                scaleratio=1,     # Ensure equal scaling
            ),
            yaxis=dict(
                visible=False,
                scaleanchor='x',  # Anchor y-axis scale to x-axis
                scaleratio=1,     # Ensure equal scaling
            ),
            # Make it auto-resize to fit the screen (important to make single mode take the full width)
            # width=1200,
            # height=600,
            # Add padding instead
            margin=dict(l=0, r=0, b=0, t=0),
            plot_bgcolor='white'
        )

        return fig

    def get_model_description_md(self, task_type="retrieval"):
        model_description_md = """
| | | |
| ---- | ---- | ---- |
"""
        if task_type == "retrieval":
            models = self.models_retrieval
        elif task_type == "clustering":
            models = self.models_clustering
        else:
            models = self.models_sts

        ct = 0
        for i, name in enumerate(models):
            one_model_md = f"[{name}]({self.model_meta[name]['link']}): {self.model_meta[name]['desc']}"
            if ct % 3 == 0:
                model_description_md += "|"
            model_description_md += f" {one_model_md} |"
            if ct % 3 == 2:
                model_description_md += "\n"
            ct += 1
        return model_description_md
