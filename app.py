from pathlib import Path
from yaml import safe_load
import gradio as gr
import os

from leaderboard import build_leaderboard_tab
from models import ModelManager
from ui import build_side_by_side_ui_anon, build_side_by_side_ui_anon_sts, build_side_by_side_ui_anon_clustering, build_side_by_side_ui_named, build_side_by_side_ui_named_sts, build_side_by_side_ui_named_clustering, build_single_model_ui, build_single_model_ui_sts, build_single_model_ui_clustering

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
        for file in elo_results_dir.glob('elo_results_*.pkl'):
            if 'clustering' in file.name:
                elo_results_file['clustering'] = file
            elif 'retrieval' in file.name:
                elo_results_file['retrieval'] = file
            elif 'sts' in file.name:
                elo_results_file['sts'] = file
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
    with gr.Tab("Retrieval", id=0):
        with gr.Tabs() as tabs_ig:
            with gr.Tab("Retrieval Arena (battle)", id=0):
                build_side_by_side_ui_anon(models)

            with gr.Tab("Retrieval Arena (side-by-side)", id=1):
                build_side_by_side_ui_named(models)

            with gr.Tab("Retrieval Playground", id=2):
                build_single_model_ui(models)

            if (elo_results_file) and ('retrieval' in elo_results_file):
                with gr.Tab("Retrieval Leaderboard", id=3):
                    build_leaderboard_tab(elo_results_file['retrieval'], leaderboard_table_file['retrieval'], task_type="Retrieval")
    
    with gr.Tab("Clustering", id=5):
        with gr.Tabs() as tabs_ie:
            with gr.Tab("Clustering Arena (battle)", id=5):
                build_side_by_side_ui_anon_clustering(models)

            with gr.Tab("Clustering Arena (side-by-side)", id=6):
                build_side_by_side_ui_named_clustering(models)

            with gr.Tab("Clustering Playground", id=7): #Direct Chat
                build_single_model_ui_clustering(models)

            if (elo_results_file) and ('clustering' in elo_results_file):
                with gr.Tab("Clustering Leaderboard", id=8):
                    build_leaderboard_tab(elo_results_file['clustering'], leaderboard_table_file['clustering'], task_type="Clustering")

    with gr.Tab("STS", id=10):
        with gr.Tabs() as tabs_vg:
            with gr.Tab("STS Arena (battle)", id=10):
                build_side_by_side_ui_anon_sts(models)

            with gr.Tab("STS Arena (side-by-side)", id=11):
                build_side_by_side_ui_named_sts(models)

            with gr.Tab("STS Playground", id=12):
                build_single_model_ui_sts(models)

            if (elo_results_file) and ('sts' in elo_results_file):
                with gr.Tab("STS Leaderboard", id=3):
                    build_leaderboard_tab(elo_results_file['sts'], leaderboard_table_file['sts'], task_type="STS")

    # with gr.Tab("About Us", id=4): 
    #     pass#build_about()

block.queue(max_size=10)
block.launch()
