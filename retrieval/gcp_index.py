import math
import json
import os
import time
from tqdm import tqdm
from google.cloud import aiplatform, storage

from .common import load_passages_from_hf
from log_utils import build_logger

logger = build_logger("index_logger", "index_logger.log")


def model_name_as_path(model_name) -> str:
    return model_name.replace("/", "__").replace(" ", "_")

MODEL_TO_INDEX_MAP = {} # for debugging with custom index names
# https://cloud.withgoogle.com/region-picker/; us-central-1 & us-east-1 are cheapest
INDEX_TO_REGION_MAP = {
    "index_stackexchange_Salesforce__SFR-Embedding-2_R": "us-central1",
    "index_stackexchange_text-embedding-004": "us-central1",
    "index_stackexchange_intfloat__e5-mistral-7b-instruct": "us-central1",
    "index_stackexchange_voyage-multilingual-2": "us-central1",
    "index_wikipedia_text-embedding-004": "us-central1",
}

class VertexIndex:
    """
    A GCP Vertex AI Vector Search wrapper.
    """
    index: aiplatform.MatchingEngineIndex = None
    endpoint: aiplatform.MatchingEngineIndexEndpoint = None
    PROJECT_ID = "contextual-research-common"
    MACHINE_TYPE = "e2-standard-16"

    def __init__(self, dim: int, model_name: str, model, corpus: str = "wikipedia", limit=None):
        model_path = model_name_as_path(model_name)
        self.index_name = MODEL_TO_INDEX_MAP.get(model_name, f"index_{corpus}_{model_path}".replace(".", "_"))
        region = INDEX_TO_REGION_MAP.get(self.index_name, "")
        if region == "":
            region = "us-east1" if corpus in ["wikipedia", "stackexchange"] else "us-central1"
        self.gcs_bucket_name = "mtebarenauscentral" if region == "us-central1" else "mtebarena"
        self.gcs_bucket_uri = f"gs://{self.gcs_bucket_name}"
        aiplatform.init(project=self.PROJECT_ID, location=region)
        self.dim = dim
        self.model = model
        # GCP filters do not allow `.` in the name, see _index_exists()
        self.index_resource_name = None
        self.deploy_index_name = None
        self.endpoint_name = "endpoint" # Reuse endpoint across indexes
        self.emb_file_path = f"emb_{corpus}_{model_path}.json"
        self.emb_folder = f"emb_{corpus}_{model_path}"
        self.endpoint_resource_name = None
        self.passages = load_passages_from_hf(corpus=corpus, limit=limit)
        self.doc_map = {str(i): doc for i, doc in enumerate(self.passages)}

    def _index_exists(self) -> bool:
        # return False
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

    def _write_embeddings_to_file(self, embeddings, indices):
        with open(self.emb_file_path, "a") as f:
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

    # def _write_embeddings(self, gpu_embedder_batch_size=32//4) -> None:#32//4) -> None:
    def _write_embeddings(self, gpu_embedder_batch_size=32*16) -> None:        
        """Batch encoding passages, then write a jsonl file."""
        if os.path.exists(self.emb_file_path):
            raise FileExistsError(f"{self.emb_file_path} already exists. Delete it before running this method.")
        logger.info(f"Writing embeddings to {self.emb_file_path} ...")

        """
        seen_ids = set()
        with open("emb_wikipedia_text-embedding-004.json_122534567101114151618", 'r') as f:
            for line in f:
                data = json.loads(line)
                if data['id'] not in seen_ids:
                    seen_ids.add(str(data['id']))
        """           
        
        n_batch = math.ceil(len(self.passages) / gpu_embedder_batch_size)
        total = 0
        for i in tqdm(range(n_batch), desc="Encoding passages"):
            print("I", i)
            
            indices = range(i * gpu_embedder_batch_size, (i + 1) * gpu_embedder_batch_size)
            """
            if all([str(index) in seen_ids for index in indices]):
                print("Skipping as all indices in seen_ids")
                continue
            elif any([str(index) in seen_ids for index in indices]):
                raise ValueError("Some indices are missing in seen_ids")
            else:
                print("All indices are missing in seen_ids", indices, len(seen_ids))
                # exit()
            """
            """
            assert len(indices) == 1
            if indices[0] in seen_ids:
                continue
            """

            batch = self.passages[i * gpu_embedder_batch_size : (i + 1) * gpu_embedder_batch_size]
            if hasattr(self.model, "encode_corpus"):
                embeddings = self.model.encode_corpus(batch, batch_size=gpu_embedder_batch_size//8)
            else:
                embeddings = self.model.encode(batch, batch_size=gpu_embedder_batch_size//8)
            total += len(embeddings)
            self._write_embeddings_to_file(embeddings.tolist(), indices)
            if i % 500 == 0 and i > 0:
                logger.info(f"Number of passages encoded: {total}")
        logger.info(f"{total} passages encoded.")

    def _upload_embedding_file(self) -> None:
        """Upload temp file to GCP storage bucket."""
        logger.info(f"Uploading {self.emb_file_path} to {self.gcs_bucket_uri}/{self.emb_folder}")
        storage_client = storage.Client()
        bucket = storage_client.bucket(self.gcs_bucket_name)
        # Include the folder name in the blob path
        blob = bucket.blob(f"{self.emb_folder}/{self.emb_file_path.split('/')[-1]}")
        blob.upload_from_filename(self.emb_file_path)

    def _create_index(self) -> None:
        """
        Create empty index and update it with embeddings.
        Reference: https://cloud.google.com/vertex-ai/docs/vector-search/configuring-indexes
        """
        self._write_embeddings()
        #exit()
        self._upload_embedding_file()
        #exit()
        logger.info(f"Creating Vector Search index {self.index_name} ...")
        self.index = aiplatform.MatchingEngineIndex.create_tree_ah_index(
            display_name=self.index_name,
            dimensions=self.dim,
            contents_delta_uri=self.gcs_bucket_uri + "/" + self.emb_folder,
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
    
    def search(self, query_embeds: list, topk=1, num_retries=5):
        """Return topk docs"""
        if self.endpoint is None:
            self.load_endpoint()

        # https://github.com/google-gemini/generative-ai-python/issues/64
        while num_retries > 0:
            try:
                response = self.endpoint.find_neighbors(
                    deployed_index_id=self.deploy_index_name,
                    queries=query_embeds,
                    num_neighbors=topk,
                )
                break
            except Exception as e:
                num_retries -= 1
                logger.error(f"Error in find_neighbors: {e}. Retries left: {num_retries}")
                time.sleep(2)

        sorted_data = sorted(response[0], key=lambda x: x.distance, reverse=True)
        docs = [self.doc_map[x.id] for x in sorted_data]
        return docs

    def cleanup(self):
        self.endpoint.delete(force=True)
        self.index.delete(sync=False)


if __name__ == '__main__':
    print(len(load_passages_from_hf(corpus='stackexchange')))