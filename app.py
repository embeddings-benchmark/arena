from pathlib import Path
from yaml import safe_load
import gradio as gr
import os

from leaderboard import build_leaderboard_tab
from models import ModelManager
from ui import build_side_by_side_ui_anon, build_side_by_side_ui_named, build_single_model_ui

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
            if 'retrieval' in file.name:
                elo_results_file['retrieval'] = file
            else:
                raise ValueError(f"Unknown file name: {file.name}")
        for file in elo_results_dir.glob('*_leaderboard.csv'):
            if 'retrieval' in file.name:
                leaderboard_table_file['retrieval'] = file
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

            if elo_results_file:
                with gr.Tab("Retrieval Leaderboard", id=3):
                    build_leaderboard_tab(elo_results_file['retrieval'], leaderboard_table_file['retrieval'])
    
    with gr.Tab("Clustering", id=5):
        with gr.Tabs() as tabs_ie:
            with gr.Tab("Clustering Arena (battle)", id=5):
                pass#build_side_by_side_ui_anony_ie(models)

            with gr.Tab("Clustering Arena (side-by-side)", id=6):
                pass#build_side_by_side_ui_named_ie(models)

            with gr.Tab("Clustering Playground", id=7): #Direct Chat
                pass#build_single_model_ui_ie(models, add_promotion_links=True)
            #if elo_results_file:
            #    with gr.Tab("Clustering Leaderboard", id=8):
            #        build_leaderboard_tab(elo_results_file['image_editing'], leaderboard_table_file['image_editing'])

    with gr.Tab("STS", id=10):
        with gr.Tabs() as tabs_vg:
            with gr.Tab("STS Arena (battle)", id=10):
                pass#build_side_by_side_ui_anony_vg(models)

            with gr.Tab("STS Arena (side-by-side)", id=11):
                pass#build_side_by_side_ui_named_vg(models)

            with gr.Tab("STS Playground", id=12): #Direct Chat
                pass#build_single_model_ui_vg(models, add_promotion_links=True)
            #if elo_results_file and 'video_generation' in elo_results_file:
            #    with gr.Tab("Video Generation Leaderboard", id=13):
            #        build_leaderboard_tab(elo_results_file['video_generation'], leaderboard_table_file['video_generation'])

    with gr.Tab("About Us", id=4): 
        pass#build_about()

block.queue(max_size=10)
block.launch()
