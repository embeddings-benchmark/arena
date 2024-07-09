import math
import json
import os
from tqdm import tqdm
from google.cloud import aiplatform, storage

from .common import load_passages_from_hf
from log_utils import build_logger

logger = build_logger("index_logger", "index_logger.log")


def model_name_as_path(model_name) -> str:
    return model_name.replace("/", "__").replace(" ", "_")


class VertexIndex:
    """
    A GCP Vertex AI Vector Search wrapper.
    """
    index: aiplatform.MatchingEngineIndex = None
    endpoint: aiplatform.MatchingEngineIndexEndpoint = None
    PROJECT_ID = "contextual-research-common"
    # us-central-1 & us-east-1 are cheapest
    # https://cloud.withgoogle.com/region-picker/
    # REGION = "us-east1" # "us-central1"
    MACHINE_TYPE = "e2-standard-16"

    def __init__(self, dim: int, model_name: str, model, corpus: str = "wikipedia", limit=None):
        region = "us-east1" if corpus == "wikipedia" else "us-central1"
        self.gcs_bucket_name = "mtebarena" if corpus == "wikipedia" else "mtebarenauscentral"
        self.gcs_bucket_uri = f"gs://{self.gcs_bucket_name}"
        aiplatform.init(project=self.PROJECT_ID, location=region)
        self.dim = dim
        self.model = model
        model_path = model_name_as_path(model_name)
        # GCP filters do not allow `.` in the name, see _index_exists()
        self.index_name = f"index_{corpus}_{model_path}".replace(".", "_")
        self.index_resource_name = None
        self.deploy_index_name = None
        # Reuse endpoint across indexes
        self.endpoint_name = "endpoint" # f"endpoint_{corpus}_{model_path}"
        self.tmp_file_path = f"tmp_{corpus}_{model_path}.json"
        self.tmp_folder = f"tmp_{corpus}_{model_path}"
        self.endpoint_resource_name = None
        self.passages = load_passages_from_hf(corpus=corpus, limit=limit)
        self.doc_map = {str(i): doc for i, doc in enumerate(self.passages)}

    def _index_exists(self) -> bool:
        logger.info(f"Checking if index {self.index_name} exists ...")
        # This fails with `google.api_core.exceptions.InvalidArgument: 400 Provided filter is not valid.` if `.` is in the name
        index_names = [
            index.resource_name
            for index in aiplatform.MatchingEngineIndex.list(
                filter=f"display_name={self.index_name}"
            )
        ]
        if len(index_names):
            self.index_resource_name = index_names[0]
            return True
        return False

    def _write_embeddings_to_tmp_file(self, embeddings, indices):
        with open(self.tmp_file_path, "a") as f:
            embeddings_formatted = [
                json.dumps(
                    {
                        "id": str(index),
                        "embedding": [str(value) for value in embedding],
                    }
                )
                + "\n"
                for index, embedding in zip(indices, embeddings)
            ]
            f.writelines(embeddings_formatted)

    def _write_embeddings(self, gpu_embedder_batch_size=32*8) -> None:
        """Batch encoding passages, then write a jsonl file."""
        if os.path.exists(self.tmp_file_path):
            os.remove(self.tmp_file_path)

        n_batch = math.ceil(len(self.passages) / gpu_embedder_batch_size)
        total = 0
        for i in tqdm(range(n_batch), desc="Encoding passages"):
            indices = range(i * gpu_embedder_batch_size, (i + 1) * gpu_embedder_batch_size)
            batch = self.passages[i * gpu_embedder_batch_size : (i + 1) * gpu_embedder_batch_size]
            if hasattr(self.model, "encode_corpus"):
                embeddings = self.model.encode_corpus(batch, batch_size=gpu_embedder_batch_size//8)
            else:
                embeddings = self.model.encode(batch, batch_size=gpu_embedder_batch_size//8)
            total += len(embeddings)
            self._write_embeddings_to_tmp_file(embeddings.tolist(), indices)
            if i % 500 == 0 and i > 0:
                logger.info(f"Number of passages encoded: {total}")
        logger.info(f"{total} passages encoded.")

    def _upload_embedding_file(self) -> None:
        """Upload temp file to GCP storage bucket."""
        logger.info(f"Uploading {self.tmp_file_path} to {self.gcs_bucket_uri}/{self.tmp_folder}")
        storage_client = storage.Client()
        bucket = storage_client.bucket(self.gcs_bucket_name)
        # Include the folder name in the blob path
        blob = bucket.blob(f"{self.tmp_folder}/{self.tmp_file_path.split('/')[-1]}")
        blob.upload_from_filename(self.tmp_file_path)

    def _create_index(self) -> None:
        """
        Create empty index and update it with embeddings.
        Reference: https://cloud.google.com/vertex-ai/docs/vector-search/configuring-indexes
        """
        self._write_embeddings()
        self._upload_embedding_file()
        logger.info(f"Creating Vector Search index {self.index_name} ...")
        self.index = aiplatform.MatchingEngineIndex.create_tree_ah_index(
            display_name=self.index_name,
            dimensions=self.dim,
            contents_delta_uri=self.gcs_bucket_uri + "/" + self.tmp_folder,
            approximate_neighbors_count=150,
            distance_measure_type="DOT_PRODUCT_DISTANCE",
            feature_norm_type="UNIT_L2_NORM",
            shard_size="SHARD_SIZE_SMALL",
            index_update_method="BATCH_UPDATE",
        )
        logger.info(
            f"Vector Search index {self.index.display_name} created with resource name {self.index.resource_name}"
        )

    def _load_index(self) -> None:
        """Load self.index if exists. Create and load index if not."""
        if self._index_exists():
            self.index = aiplatform.MatchingEngineIndex(index_name=self.index_resource_name)
            logger.info(f"Vector Search index {self.index.display_name} exists with resource name {self.index.resource_name}")
            return
        print(f"Index does not exist. Creating {self.index_name}")
        self._create_index()
    
    def _endpoint_exists(self) -> bool:
        endpoint_names = [
            endpoint.resource_name
            for endpoint in aiplatform.MatchingEngineIndexEndpoint.list(
                filter=f"display_name={self.endpoint_name}"
            )
        ]
        if len(endpoint_names):
            self.endpoint_resource_name = endpoint_names[0]
            return True
        return False
    
    def _endpoint_deployed(self) -> bool:
        index_endpoints = [
            (deployed_index.index_endpoint, deployed_index.deployed_index_id)
            for deployed_index in self.index.deployed_indexes
        ]

        if len(index_endpoints):
            self.index_endpoint_name = index_endpoints[0][0]
            self.deploy_index_name = index_endpoints[0][1]
            return True
        return False

    def load_endpoint(self) -> None:
        """Load a public endpoint if exists. Create and load endpoint if not."""
        if self.index is None:
            self._load_index()

        if self._endpoint_exists():
            self.endpoint = aiplatform.MatchingEngineIndexEndpoint(
                index_endpoint_name=self.endpoint_resource_name
            )
            logger.info(
                f"Vector Search index endpoint {self.endpoint.display_name} exists with resource name {self.endpoint.resource_name}"
            )
        else: 
            logger.info(f"Creating Vector Search index endpoint {self.endpoint_name} ...")
            self.endpoint = aiplatform.MatchingEngineIndexEndpoint.create(
                display_name=self.endpoint_name, public_endpoint_enabled=True
            )
            self.endpoint_resource_name = self.endpoint.resource_name

        if self._endpoint_deployed():
            self.endpoint = aiplatform.MatchingEngineIndexEndpoint(
                index_endpoint_name=self.index_endpoint_name
            )
            return

        ## Synchronous call. This could take up to 30 minutes.
        logger.info(f"Deploying Vector Search index {self.index.display_name}...")
        self.deploy_index_name = "endpoint_" + self.endpoint_resource_name.split("/")[-1]
        self.endpoint = self.endpoint.deploy_index(
            index=self.index,
            deployed_index_id=self.deploy_index_name,
            display_name=self.index_name,
            machine_type=self.MACHINE_TYPE,
            min_replica_count=1,
            max_replica_count=1,
        )
        logger.info(
            f"Vector Search index {self.index.display_name} is deployed at endpoint {self.endpoint.display_name}"
        )
    
    def search(self, query_embeds: list, topk=1):
        """Return topk docs"""
        if self.endpoint is None:
            self.load_endpoint()

        response = self.endpoint.find_neighbors(
            deployed_index_id=self.deploy_index_name,
            queries=query_embeds,
            num_neighbors=topk,
        )

        sorted_data = sorted(response[0], key=lambda x: x.distance, reverse=True)
        docs = [self.doc_map[x.id] for x in sorted_data]
        return docs

    def cleanup(self):
        self.endpoint.delete(force=True)
        self.index.delete(sync=False)


if __name__ == '__main__':
    print(len(load_passages_from_hf(corpus='stackexchange')))