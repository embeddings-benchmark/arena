from pathlib import Path
from yaml import safe_load
import gradio as gr
import os

from leaderboard import build_leaderboard_tab
from models import ModelManager
from ui import build_side_by_side_ui_anon, build_side_by_side_ui_anon_sts, build_side_by_side_ui_anon_clustering, build_side_by_side_ui_named, build_side_by_side_ui_named_sts, build_side_by_side_ui_named_clustering, build_single_model_ui, build_single_model_ui_sts, build_single_model_ui_clustering

from arena_elo.elo_analysis import load_results


# download the videos 
from huggingface_hub import hf_hub_url
for file_to_download in ["sts_explanation.mp4", "clustering_explanation.mp4"]:
    file_url = hf_hub_url(repo_id="mteb/arena-videos", repo_type="dataset", endpoint=None, filename=file_to_download)
    # download it to videos/ folder using wget
    os.system(f"wget {file_url} -O videos/{file_to_download}")



acknowledgment_md = """
### Acknowledgment
We thank X, Y, Z, [Contextual AI](https://contextual.ai/) and [Hugging Face](https://huggingface.co/) for their generous sponsorship. If you'd like to sponsor us, please get in [touch](mailto:n.muennighoff@gmail.com).

<div class="sponsor-image-about" style="display: flex; align-items: center; gap: 10px;">
    <a href="https://contextual.ai/">
        <img src="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQd4EDMoZLFRrIjVBrSXOQYGcmvUJ3kL4U2usvjuKPla-LoRTZtLzFnb_Cu5tXzRI7DNBo&usqp=CAU" width="60" height="55" style="padding: 10px;">
    </a>
    <a href="https://huggingface.co">
        <img src="https://raw.githubusercontent.com/embeddings-benchmark/mteb/main/docs/images/hf_logo.png" width="60" height="55" style="padding: 10px;">
    </a>
</div>

We thank [Chatbot Arena](https://chat.lmsys.org/), [Vision Arena](https://huggingface.co/spaces/WildVision/vision-arena) and [GenAI-Arena](https://huggingface.co/spaces/TIGER-Lab/GenAI-Arena) for inspiration.
"""

ELO_RESULTS_DIR = os.getenv("ELO_RESULTS_DIR", "./results/latest")
MODEL_META_PATH = "model_meta.yml"
# Debugging
MODEL_META_PATH = "model_meta_debug.yml"
with open(MODEL_META_PATH, 'r', encoding='utf-8') as f:
    model_meta = safe_load(f)
models = ModelManager(model_meta)

def load_elo_results(elo_results_dir):
    from collections import defaultdict
    elo_results_file = defaultdict(lambda: None)
    leaderboard_table_file = defaultdict(lambda: None)
    if elo_results_dir is not None:
        elo_results_dir = Path(elo_results_dir)
        elo_results_file = {}
        leaderboard_table_file = {}
        for folder in elo_results_dir.glob('elo_results_*'):
            if 'clustering' in file.name:
                elo_results_file['clustering'] = load_results(folder)
            elif 'retrieval' in file.name:
                elo_results_file['retrieval'] = load_results(folder)
            elif 'sts' in file.name:
                elo_results_file['sts'] = load_results(folder)
            else:
                raise ValueError(f"Unknown file name: {file.name}")
        for file in elo_results_dir.glob('*_leaderboard.csv'):
            if 'clustering' in file.name:
                leaderboard_table_file['clustering'] = file
            elif 'retrieval' in file.name:
                leaderboard_table_file['retrieval'] = file
            elif 'sts' in file.name:
                leaderboard_table_file['sts'] = file
            else:
                raise ValueError(f"Unknown file name: {file.name}")
            
    return elo_results_file, leaderboard_table_file

elo_results_file, leaderboard_table_file = load_elo_results(ELO_RESULTS_DIR)

head_js = """
<script src="https://cdnjs.cloudflare.com/ajax/libs/html2canvas/1.4.1/html2canvas.min.js"></script>
"""

with gr.Blocks(title="MTEB Arena", head=head_js) as block:
    with gr.Tab("🔎 Retrieval", id=0):
        with gr.Tabs() as tabs_ig:
            with gr.Tab("⚔️ Arena (battle)", id=0):
                build_side_by_side_ui_anon(models)

            with gr.Tab("⚔️ Arena (side-by-side)", id=1):
                build_side_by_side_ui_named(models)

            with gr.Tab("💧 Single", id=2):
                build_single_model_ui(models)

            if (elo_results_file) and ('retrieval' in elo_results_file):
                with gr.Tab("🏆 Leaderboard", id=3):
                    build_leaderboard_tab(elo_results_file['retrieval'], leaderboard_table_file['retrieval'], task_type="Retrieval")

    with gr.Tab("✨ Clustering", id=5):
        with gr.Tabs() as tabs_ie:
            with gr.Tab("⚔️ Arena (battle)", id=5):
                build_side_by_side_ui_anon_clustering(models)

            with gr.Tab("⚔️ Arena (side-by-side)", id=6):
                build_side_by_side_ui_named_clustering(models)

            with gr.Tab("💧 Single", id=7): #Direct Chat
                build_single_model_ui_clustering(models)

            if (elo_results_file) and ('clustering' in elo_results_file):
                with gr.Tab("🏆 Leaderboard", id=8):
                    build_leaderboard_tab(elo_results_file['clustering'], leaderboard_table_file['clustering'], task_type="Clustering")

    with gr.Tab("☘️ STS", id=10):
        with gr.Tabs() as tabs_vg:
            with gr.Tab("⚔️ Arena (battle)", id=10):
                build_side_by_side_ui_anon_sts(models)

            with gr.Tab("⚔️ Arena (side-by-side)", id=11):
                build_side_by_side_ui_named_sts(models)

            with gr.Tab("💧 Single", id=12):
                build_single_model_ui_sts(models)

            if (elo_results_file) and ('sts' in elo_results_file):
                with gr.Tab("🏆 Leaderboard", id=3):
                    build_leaderboard_tab(elo_results_file['sts'], leaderboard_table_file['sts'], task_type="STS")

    gr.Markdown(acknowledgment_md, elem_id="ack_markdown")

    # with gr.Tab("About Us", id=4): 
    #     pass#build_about()

block.queue(max_size=10)
block.launch(share=True)
