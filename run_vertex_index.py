"""Script for testing the VertexIndex code and setup the vector search index."""
import sys
from yaml import safe_load
from models import ModelManager

MODEL_META_PATH = "model_meta.yml"
with open(MODEL_META_PATH, 'r', encoding='utf-8') as f:
    model_meta = safe_load(f)

## Enable GCP Vertex index
models = ModelManager(model_meta, use_gcp_index=True)

model_name = sys.argv[1] # "intfloat/multilingual-e5-small"
corpus = sys.argv[2] # "wikipedia"

"""For uploading already finished embedding files
def model_name_as_path(model_name) -> str:
    return model_name.replace("/", "__").replace(" ", "_")
path = model_name_as_path(model_name)
from google.cloud import aiplatform, storage
storage_client = storage.Client()
bucket = storage_client.bucket("mtebarenauscentral")
bucket = storage_client.bucket("mtebarena")
# Include the folder name in the blob path
blob = bucket.blob(f"emb_{corpus}_{path}/emb_{corpus}_{path}.json")
blob.upload_from_filename(f"emb_{corpus}_{path}.json")
dim = model_meta['model_meta'][model_name]["dim"]
#aiplatform.init(project="contextual-research-common", location="us-central1")
#aiplatform.init(project="contextual-research-common", location="us-east1")
#index = aiplatform.MatchingEngineIndex.create_tree_ah_index(display_name=f"index_{corpus}_{path}", dimensions=dim, contents_delta_uri=f"gs://mtebarenauscentral/tmp_{corpus}_{path}", approximate_neighbors_count=150, distance_measure_type="DOT_PRODUCT_DISTANCE", feature_norm_type="UNIT_L2_NORM", shard_size="SHARD_SIZE_SMALL", index_update_method="BATCH_UPDATE",)
#index = aiplatform.MatchingEngineIndex.create_tree_ah_index(display_name=f"index_{corpus}_{path}", dimensions=dim, contents_delta_uri=f"gs://mtebarena/tmp_{corpus}_{path}", approximate_neighbors_count=150, distance_measure_type="DOT_PRODUCT_DISTANCE", feature_norm_type="UNIT_L2_NORM", shard_size="SHARD_SIZE_SMALL", index_update_method="BATCH_UPDATE",)
#index_name = f"index_{corpus}_{path}"
#index_names = [index.resource_name for index in aiplatform.MatchingEngineIndex.list()]#filter=f"display_name={index_name}")]
if len(index_names):
    print("index_names", index_names)
    index_resource_name = index_names[0]
    print("INDEX RESOURCE NAME", index_resource_name)
    exit()
#index_resource_name="1910731306749132800"
index = aiplatform.MatchingEngineIndex(index_name=index_resource_name)
endpoint = aiplatform.MatchingEngineIndexEndpoint.create(display_name=f"endpoint_{corpus}_{path}", public_endpoint_enabled=True)
endpoint_resource_name = endpoint.resource_name
endpoint = endpoint.deploy_index(index=index, deployed_index_id="endpoint_" + endpoint_resource_name.split("/")[-1], display_name=f"index_{corpus}_{path}", machine_type="e2-standard-16", min_replica_count=1, max_replica_count=1,)
"""
print(models.retrieve(query="Where is Japan?", model_name=model_name, corpus=corpus))