import os
import time
import uuid
import shutil
import random
import datetime
import atexit
from functools import partial

import torch
import gradio as gr

from log_utils import build_logger, store_data_in_hub

LOGDIR = os.getenv("LOGDIR", "./MTEB-Arena-logs/vote_log")

DEFAULT_MODEL_A = "GritLM/GritLM-7B"
DEFAULT_MODEL_B = "BAAI/bge-large-en-v1.5"

info_txt = "🎉 Thanks for voting! Your vote shapes the leaderboard, please vote RESPONSIBLY."

# loggers for side-by-side and battle
retrieval_logger = build_logger("gradio_retrieval", "gradio_retrieval.log")
clustering_logger = build_logger("gradio_clustering", "gradio_clustering.log")
sts_logger = build_logger("gradio_sts", "gradio_sts.log")


def save_tensor(tensor, file_path):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    torch.save(tensor, file_path)


def cleanup_dirs(*dir_paths):
    for dir_path in dir_paths:
        try:
            if os.path.exists(dir_path) and os.path.isdir(dir_path):
                shutil.rmtree(dir_path)
        except Exception as e:
            print(f"Error deleting directory {dir_path}: {e}")

def get_ip(request: gr.Request):
    if request:
        if "cf-connecting-ip" in request.headers:
            ip = request.headers["cf-connecting-ip"] or request.client.host
        else:
            ip = request.client.host
    else:
        ip = ""
    return ip

def clear_history(): return None, "", None
def clear_history_sts(): return None, "", "", "", None
def clear_history_clustering(): return None, "", 1, None
def clear_history_side_by_side(state0, state1): 
    cleanup_dirs(os.path.dirname(state0.query_embed_file_path), os.path.dirname(state1.query_embed_file_path))
    return None, None, "", None, None, gr.DownloadButton(visible=False), gr.DownloadButton(visible=False)
def clear_history_side_by_side_anon(state0, state1):
    cleanup_dirs(os.path.dirname(state0.query_embed_file_path), os.path.dirname(state1.query_embed_file_path))
    return None, None, "", None, None, gr.Markdown("", visible=False), gr.Markdown("", visible=False), gr.DownloadButton(visible=False), gr.DownloadButton(visible=False)
def clear_history_side_by_side_anon_sts():
    return None, None, "", "", "",  None, None, gr.Markdown("", visible=False), gr.Markdown("", visible=False)
def clear_history_side_by_side_anon_clustering():
    return None, None, "", 1, None, None, gr.Markdown("", visible=False), gr.Markdown("", visible=False)

def disable_btns(i=6, visible=True): return (gr.update(interactive=False, visible=visible),) * i
def enable_btns(i=6, visible=True): return (gr.update(interactive=True, visible=visible),) * i

def enable_btns_clustering(state0):
    if (hasattr(state0, "prompts")) and (len(state0.prompts) >= 3): return enable_btns(10)
    return enable_btns(5) + disable_buttons_side_by_side(5)

def disable_buttons_side_by_side(i=6):
    return tuple(gr.update(visible=i>=4, interactive=False) for i in range(i))

def vote_last_response(vote_type, state0, state1, model_selector0, model_selector1, request: gr.Request):
    if vote_type != "share":
        gr.Info(info_txt)
    retrieval_logger.info(f"{vote_type} (named). ip: {get_ip(request)}")

    data = {
        "tstamp": round(time.time(), 4),
        "task_type": "retrieval",
        "type": vote_type,
        "models": [model_selector0, model_selector1],
        "ip": get_ip(request),
        **state0.dict(prefix="0"),
        **state1.dict(prefix="1")
    }
    # if models are anonymous, send to battle, otherwise side-by-side
    if model_selector0 in ["", None] and model_selector1 in ["", None]:
        store_data_in_hub(data, "retrieval_battle")
    else:
        store_data_in_hub(data, "retrieval_side_by_side")

    if vote_type == "share": return

    
 
    return_state = ("Press 🎲 New Round to start over 👇 (Note: Your vote shapes the leaderboard, please vote RESPONSIBLY!)",) + disable_btns(4) 
    if model_selector0 == "":
        return_state  = return_state + (gr.Markdown(f"### Model A: {state0.model_name}", visible=True), gr.Markdown(f"### Model B: {state1.model_name}", visible=True))
    else: 
        return_state = return_state + (gr.Markdown(state0.model_name, visible=True), gr.Markdown(state1.model_name, visible=True))
    
    if os.path.exists(state0.query_embed_file_path) and os.path.exists(state1.query_embed_file_path):
        download_a_btn = gr.update(
            label = "📥 Download embedding for model A", value=state0.query_embed_file_path, visible=True
        )
        download_b_btn = gr.update(
            label = "📥 Download embedding for model B", value=state1.query_embed_file_path, visible=True
        )
        return_state = return_state + (download_a_btn, download_b_btn)
    return return_state

def vote_last_response_sts(vote_type, state0, state1, model_selector0, model_selector1, request: gr.Request):
    if vote_type != "share":
        gr.Info(info_txt)
    sts_logger.info(f"{vote_type} (named). ip: {get_ip(request)}")

    data = {
        "tstamp": round(time.time(), 4),
        "task_type": "sts",
        "type": vote_type,
        "models": [model_selector0, model_selector1],
        "ip": get_ip(request),
        **state0.dict(prefix="0"),
        **state1.dict(prefix="1")
    }
    # if models are anonymous, send to battle, otherwise side-by-side
    if model_selector0 in ["", None] and model_selector1 in ["", None]:
        store_data_in_hub(data, "sts_battle")
    else:
        store_data_in_hub(data, "sts_side_by_side")

    if vote_type == "share": return

    if model_selector0 == "":
        return disable_btns(4) + (gr.Markdown(f"### Model A: {state0.model_name}", visible=True), gr.Markdown(f"### Model B: {state1.model_name}", visible=True))
    return disable_btns(4) + (gr.Markdown(state0.model_name, visible=True), gr.Markdown(state1.model_name, visible=True))

def vote_last_response_clustering(vote_type, state0, state1, model_selector0, model_selector1, request: gr.Request):
    if vote_type != "share":
        gr.Info(info_txt)
    clustering_logger.info(f"{vote_type} (named). ip: {get_ip(request)}")

    data = {
        "tstamp": round(time.time(), 4),
        "task_type": "clustering",
        "type": vote_type,
        "models": [model_selector0, model_selector1],
        "ip": get_ip(request),
        **state0.dict(prefix="0"),
        **state1.dict(prefix="1")
    }
    # if models are anonymous, send to battle, otherwise side-by-side
    if model_selector0 in ["", None] and model_selector1 in ["", None]:
        store_data_in_hub(data, "clustering_battle")
    else:
        store_data_in_hub(data, "clustering_side_by_side")


    if vote_type == "share": return

    if model_selector0 == "":
        return disable_btns(5, visible=False) + disable_btns(4) + (gr.Markdown(f"### Model A: {state0.model_name}", visible=True), gr.Markdown(f"### Model B: {state1.model_name}", visible=True))
    return disable_btns(5, visible=False) + disable_btns(4) + (gr.Markdown(state0.model_name, visible=True), gr.Markdown(state1.model_name, visible=True))

def vote_last_response_single(vote_type, state, model_selector, request: gr.Request):
    gr.Info(info_txt)
    retrieval_logger.info(f"{vote_type} (named). ip: {get_ip(request)}")

    data = {
        "tstamp": round(time.time(), 4),
        "task_type": "retrieval",
        "type": vote_type,
        "models": model_selector,
        "ip": get_ip(request),
        **state.dict()
    }
    store_data_in_hub(data, "retrieval_single_choice")

    return ("",) + disable_btns(3)

def vote_last_response_single_sts(vote_type, state, model_selector, request: gr.Request):
    gr.Info(info_txt)
    sts_logger.info(f"{vote_type} (named). ip: {get_ip(request)}")

    data = {
        "tstamp": round(time.time(), 4),
        "task_type": "sts",
        "type": vote_type,
        "models": model_selector,
        "ip": get_ip(request),
        **state.dict()
    }
    store_data_in_hub(data, "sts_single_choice")

    return disable_btns(3)

def vote_last_response_single_clustering(vote_type, state, model_selector, request: gr.Request):
    gr.Info(info_txt)
    clustering_logger.info(f"{vote_type} (named). ip: {get_ip(request)}")

    data = {
        "tstamp": round(time.time(), 4),
        "task_type": "clustering",
        "type": vote_type,
        "models": model_selector,
        "ip": get_ip(request),
        **state.dict()
    }
    store_data_in_hub(data, "clustering_single_choice")

    return disable_btns(3)
    #return disable_btns(5, visible=False) + disable_btns(3)

def get_conv_log_filename():
    t = datetime.datetime.now()
    name = os.path.join(LOGDIR, f"{t.year}-{t.month:02d}-{t.day:02d}-conv.json")
    return name

class RetrievalState:
    def __init__(self, model_name):
        self.conv_id = uuid.uuid4().hex
        self.model_name = model_name
        self.prompt = ""
        self.corpus = ""
        self.output = ""
        self.query_embed_file_path = ""

    def dict(self, prefix: str = None):
        if prefix is None:
            return {"conv_id": self.conv_id, "model_name": self.model_name, "prompt": self.prompt, "output": self.output, "corpus": self.corpus}
        else:
            return {f"{prefix}_conv_id": self.conv_id, f"{prefix}_model_name": self.model_name, f"{prefix}_prompt": self.prompt, f"{prefix}_output": self.output, f"{prefix}_corpus": self.corpus}




def retrieve_side_by_side(gen_func, state0, state1, text, corpus, model_name0, model_name1, request: gr.Request):
    if not text: raise gr.Warning("Query cannot be empty.")
    state0, state1 = RetrievalState(model_name0), RetrievalState(model_name1)
    ip = get_ip(request)
    retrieval_logger.info(f"Retrieval. ip: {ip}")
    start_tstamp = time.time()
    retrieved_txt0, retrieved_txt1, model_name0, model_name1, query_embed0, query_embed1 = gen_func(text, corpus, model_name0, model_name1)
    state0.prompt, state1.prompt = text, text
    state0.corpus, state1.corpus = corpus, corpus
    state0.output, state1.output = retrieved_txt0, retrieved_txt1
    state0.model_name, state1.model_name = model_name0, model_name1
    
    if query_embed0 is not None and query_embed1 is not None:
        state0.query_embed_file_path = f"{state0.conv_id}_retrieve_side_by_side/model0_query_embedding.pth"
        state1.query_embed_file_path = f"{state1.conv_id}_retrieve_side_by_side/model1_query_embedding.pth"

        # Save tensors
        save_tensor(query_embed0, state0.query_embed_file_path)
        save_tensor(query_embed1, state1.query_embed_file_path)

        # Register cleanup function
        atexit.register(cleanup_dirs, os.path.dirname(state0.query_embed_file_path), os.path.dirname(state1.query_embed_file_path))


    
    yield state0, state1, retrieved_txt0, retrieved_txt1, \
        gr.Markdown(f"### Model A: {model_name0}", visible=False), gr.Markdown(f"### Model B: {model_name1}", visible=False)
    
    finish_tstamp = time.time()
    
    data = {
        "tstamp": round(finish_tstamp, 4),
        "task_type": "retrieval",
        "type": "chat",
        "model": model_name0,
        "gen_params": {},
        "start": round(start_tstamp, 4),
        "finish": round(finish_tstamp, 4),
        "ip": get_ip(request),
        **state0.dict()
    }
    store_data_in_hub(data, "retrieval_individual")

    data = {
        "tstamp": round(finish_tstamp, 4),
        "task_type": "retrieval",
        "type": "chat",
        "model": model_name1,
        "gen_params": {},
        "start": round(start_tstamp, 4),
        "finish": round(finish_tstamp, 4),
        "ip": get_ip(request),
        **state1.dict()
    }
    store_data_in_hub(data, "retrieval_individual")

def retrieve(gen_func, state, text, corpus, model_name, request: gr.Request):
    if not text: raise gr.Warning("Query cannot be empty.")
    if not model_name: raise gr.Warning("Model name cannot be empty.")
    state = RetrievalState(model_name)
    ip = get_ip(request)
    retrieval_logger.info(f"Retrieval. ip: {ip}")
    start_tstamp = time.time()
    retrieved_txt = gen_func(text, corpus, model_name)
    state.prompt = text
    state.corpus = corpus
    state.output = retrieved_txt
    state.model_name = model_name

    yield state, retrieved_txt
    
    finish_tstamp = time.time()

    data = {
        "tstamp": round(finish_tstamp, 4),
        "task_type": "retrieval",
        "type": "chat",
        "model": model_name,
        "gen_params": {},
        "start": round(start_tstamp, 4),
        "finish": round(finish_tstamp, 4),
        "ip": get_ip(request),
        **state.dict()
    }
    store_data_in_hub(data, "retrieval_individual")

def check_input_retrieval(txt):
    if not(txt): raise gr.Warning("Query cannot be empty.")

def build_side_by_side_ui_anon(models):
    notice_markdown = """
# ⚔️ MTEB Arena: Retrieval 🔎
## 📜 Rules
- Send any query to two anonymous models and vote which retrieves the better document.
- You can choose a corpus for the models to retrieve from: [Wikipedia](https://huggingface.co/datasets/mteb/arena-wikipedia-7-15-24), [arXiv](https://huggingface.co/datasets/mteb/arena-arxiv-7-2-24) or [Stack Exchange](https://huggingface.co/datasets/mteb/arena-stackexchange).
- Stack Exchange is currently in beta testing with limited models.

## 👇 Vote now!
"""

    state0 = gr.State()
    state1 = gr.State()
    gen_func = partial(retrieve_side_by_side, models.retrieve_parallel)
    
    gr.Markdown(notice_markdown, elem_id="notice_markdown")

    with gr.Group(elem_id="share-region-anon"):
        with gr.Accordion("🔍 Expand to see all Arena players", open=False):
            model_description_md = models.get_model_description_md()
            gr.Markdown(model_description_md, elem_id="model_description_markdown")
        with gr.Row():
            with gr.Column():
                chatbot_left = gr.Chatbot(
                    label="Model A",
                    elem_id="chatbot",
                    height=550,
                    show_copy_button=True,
                )
            with gr.Column():
                chatbot_right = gr.Chatbot(
                    label="Model B",
                    elem_id="chatbot",
                    height=550,
                    show_copy_button=True,
                )

        with gr.Row():
            with gr.Column():
                model_selector_left = gr.Markdown("", visible=False)
            with gr.Column():
                model_selector_right = gr.Markdown("", visible=False)

        with gr.Row():
            leftvote_btn = gr.Button(
                value="👈  A is better", visible=False, interactive=False
            )
            rightvote_btn = gr.Button(
                value="👉  B is better", visible=False, interactive=False
            )
            tie_btn = gr.Button(value="🤝  Tie", visible=False, interactive=False)
            bothbad_btn = gr.Button(
                value="👎  Both are bad", visible=False, interactive=False
            )

            download_a_btn = gr.DownloadButton(
                value="📥 Download embedding for model A", visible=False
            )
            download_b_btn = gr.DownloadButton(
                value="📥 Download embedding for model B", visible=False
            )

    with gr.Row():
        textbox = gr.Textbox(
            label="Query",
            show_label=True,
            placeholder="👉 Enter text and press ENTER",
            container=True,
            elem_id="input_box",
        )
        corpus = gr.Dropdown(
            label="Corpus",
            choices=["wikipedia", "arxiv", "stackexchange"],
            value="wikipedia",
            interactive=True,
            show_label=True,
            container=True,
            scale=0,
        )        
        send_btn = gr.Button(value="Send", variant="primary", scale=0)
        draw_btn = gr.Button(value="🎲 Random sample", variant="primary", scale=0)

    with gr.Row():
        clear_btn = gr.Button(value="🎲 New Round", interactive=False)
        share_btn = gr.Button(value="📷  Share")

    gr.Examples(
        examples=[
            ["Which test was devised to determine whether robots can think?", "wikipedia"],
            ["I am looking for a paper to help me deal with data constraints as I scale up my language model.", "arxiv"],
            ["What's the purpose of `if __name__ == '__main__'`?", "stackexchange"],
            ["Find me a planet near α 23h 06m 29.368s and δ −05° 02′ 29.04″ that could be habitable.", "wikipedia"],
            ["I am searching for a good and large-scale benchmark for testing the performance of text embeddings.", "arxiv"],
            ["In English when to use –, — vs -?", "stackexchange"],
            ["Who wrote about a guide more popular than the Encyclopedia Galactica?", "wikipedia"],
            ["Please find me the paper on training code large language models to follow instructions via git commits.", "arxiv"],
            ["Why did the machines in The Matrix keep humans around?", "stackexchange"],
        ],
        inputs=[textbox, corpus],
    )

    btn_list = [leftvote_btn, rightvote_btn, tie_btn, bothbad_btn, clear_btn]

    draw_btn.click(
        models.retrieve_draw,
        inputs=None,
        outputs=[textbox, corpus],
        api_name="draw_btn_anon"
    )

    textbox.submit(
        check_input_retrieval,
        inputs=textbox,
        outputs=None,
    ).success(
        partial(disable_buttons_side_by_side, 4),
        inputs=None,
        outputs=[textbox, corpus, send_btn, draw_btn],
    ).then(
        gen_func,
        inputs=[state0, state1, textbox, corpus, model_selector_left, model_selector_right],
        outputs=[state0, state1, chatbot_left, chatbot_right, model_selector_left, model_selector_right],
        api_name="submit_btn_anon"
    ).then(
        enable_btns,
        inputs=None,
        outputs=btn_list
    )

    send_btn.click(
        check_input_retrieval,
        inputs=textbox,
        outputs=None,
    ).success(        
        partial(disable_buttons_side_by_side, 4),
        inputs=None,
        outputs=[textbox, corpus, send_btn, draw_btn],
    ).then(
        gen_func,
        inputs=[state0, state1, textbox, corpus, model_selector_left, model_selector_right],
        outputs=[state0, state1, chatbot_left, chatbot_right, model_selector_left, model_selector_right],
        api_name="send_btn_anon"
    ).then(
        enable_btns,
        inputs=None,
        outputs=btn_list
    )

    clear_btn.click(
        clear_history_side_by_side_anon,
        inputs=[state0, state1],
        outputs=[state0, state1, textbox, chatbot_left, chatbot_right, model_selector_left, model_selector_right, download_a_btn, download_b_btn],
        api_name="clear_btn_anon"
    ).then(
        disable_buttons_side_by_side,
        inputs=None,
        outputs=btn_list
    ).then(
        partial(enable_btns, 4),
        inputs=None,
        outputs=[textbox, corpus, send_btn, draw_btn],
    )

    dummy_left_model = gr.State("")
    dummy_right_model = gr.State("")
    leftvote_btn.click(
        partial(vote_last_response, "leftvote"),
        inputs=[state0, state1, dummy_left_model, dummy_right_model],
        outputs=[textbox, leftvote_btn, rightvote_btn, tie_btn, bothbad_btn, model_selector_left, model_selector_right, download_a_btn, download_b_btn]
    )
    rightvote_btn.click(
        partial(vote_last_response, "rightvote"),
        inputs=[state0, state1, dummy_left_model, dummy_right_model],
        outputs=[textbox, leftvote_btn, rightvote_btn, tie_btn, bothbad_btn, model_selector_left, model_selector_right, download_a_btn, download_b_btn]
    )
    tie_btn.click(
        partial(vote_last_response, "tievote"),
        inputs=[state0, state1, dummy_left_model, dummy_right_model],
        outputs=[textbox, leftvote_btn, rightvote_btn, tie_btn, bothbad_btn, model_selector_left, model_selector_right, download_a_btn, download_b_btn]
    )
    bothbad_btn.click(
        partial(vote_last_response, "bothbadvote"),
        inputs=[state0, state1, dummy_left_model, dummy_right_model],
        outputs=[textbox, leftvote_btn, rightvote_btn, tie_btn, bothbad_btn, model_selector_left, model_selector_right, download_a_btn, download_b_btn]
    )

    share_js = """
function (a, b, c, d) {
    const captureElement = document.querySelector('#share-region-anon');
    html2canvas(captureElement)
        .then(canvas => {
            canvas.style.display = 'none'
            document.body.appendChild(canvas)
            return canvas
        })
        .then(canvas => {
            const image = canvas.toDataURL('image/png')
            const a = document.createElement('a')
            a.setAttribute('download', 'mteb-arena.png')
            a.setAttribute('href', image)
            a.click()
            canvas.remove()
        });
    return [a, b, c, d];
}
"""
    share_btn.click(
        partial(vote_last_response, "share"),
        inputs=[state0, state1, model_selector_left, model_selector_right],
        outputs=[],
        js=share_js
    )

def build_side_by_side_ui_named(models):
    notice_markdown = """
# ⚔️ MTEB Arena: Retrieval 🔎

## 📜 Rules
- Send any query to two anonymous models and vote which retrieves the better document.
- You can choose a corpus for the models to retrieve from: [Wikipedia](https://huggingface.co/datasets/mteb/arena-wikipedia-7-15-24) or [arXiv](https://huggingface.co/datasets/mteb/arena-arxiv-7-2-24). ([Stack Exchange](https://huggingface.co/datasets/mteb/arena-stackexchange) coming soon!)

## 👇 Choose two models & vote now!
"""
    model_list = models.models_retrieval

    state0 = gr.State()
    state1 = gr.State()
    gen_func = partial(retrieve_side_by_side, models.retrieve_parallel)
    gr.Markdown(notice_markdown, elem_id="notice_markdown")

    with gr.Group(elem_id="share-region-named"):
        with gr.Row():
            with gr.Column():
                model_selector_left = gr.Dropdown(
                    choices=model_list,
                    value=DEFAULT_MODEL_A,
                    interactive=True,
                    show_label=False,
                    container=False,
                )
            with gr.Column():
                model_selector_right = gr.Dropdown(
                    choices=model_list,
                    value=DEFAULT_MODEL_B,
                    interactive=True,
                    show_label=False,
                    container=False,
                )
        with gr.Row():
            with gr.Accordion("🔍 Expand to see all model descriptions", open=False):
                model_description_md = models.get_model_description_md()
                gr.Markdown(model_description_md, elem_id="model_description_markdown")

        with gr.Row():
            with gr.Column():
                chatbot_left = gr.Chatbot(
                    label="Model A",
                    elem_id="chatbot",
                    height=550,
                    show_copy_button=True,
                )
            with gr.Column():
                chatbot_right = gr.Chatbot(
                    label="Model B",
                    elem_id="chatbot",
                    height=550,
                    show_copy_button=True,
                )
        with gr.Row():
            leftvote_btn = gr.Button(
                value="👈  A is better", visible=False, interactive=False
            )
            rightvote_btn = gr.Button(
                value="👉  B is better", visible=False, interactive=False
            )
            tie_btn = gr.Button(value="🤝  Tie", visible=False, interactive=False)
            bothbad_btn = gr.Button(
                value="👎  Both are bad", visible=False, interactive=False
            )

            download_a_btn = gr.DownloadButton(
                value="📥 Download embedding for model A", visible=False
            )
            download_b_btn = gr.DownloadButton(
                value="📥 Download embedding for model B", visible=False
            )


    with gr.Row():
        textbox = gr.Textbox(
            label="Query",
            show_label=True,
            placeholder="👉 Enter text and press ENTER",
            container=True,
            elem_id="input_box",
        )
        corpus = gr.Dropdown(
            label="Corpus",
            choices=["wikipedia", "arxiv"], # "stackexchange"
            value="wikipedia",
            interactive=True,
            show_label=True,
            container=True,
            scale=0,
        )
        send_btn = gr.Button(value="Send", variant="primary", scale=0)
        draw_btn = gr.Button(value="🎲 Random sample", variant="primary", scale=0)

    with gr.Row():
        clear_btn = gr.Button(value="🗑️  Clear history", interactive=False)
        share_btn = gr.Button(value="📷  Share")

    gr.Examples(
        examples=[
            ["Which test was devised to determine whether robots can think?", "wikipedia"],
            ["I am looking for a paper to help me deal with data constraints as I scale up my language model.", "arxiv"],
#            ["What's the purpose of `if __name__ == '__main__'`?", "stackexchange"],
            ["Find me a planet near α 23h 06m 29.368s and δ −05° 02′ 29.04″ that could be habitable.", "wikipedia"],
            ["I am searching for a good and large-scale benchmark for testing the performance of text embeddings.", "arxiv"],
#            ["In English when to use –, — vs -?", "stackexchange"],
            ["Who wrote about a guide more popular than the Encyclopedia Galactica?", "wikipedia"],
            ["Please find me the paper on training code large language models to follow instructions via git commits.", "arxiv"],
#            ["Why did the machines in The Matrix keep humans around?", "stackexchange"],
        ],
        inputs=[textbox, corpus],
    )
    
    btn_list = [leftvote_btn, rightvote_btn, tie_btn, bothbad_btn, clear_btn]

    draw_btn.click(
        partial(models.retrieve_draw, ["wikipedia", "arxiv"]),
        inputs=None,
        outputs=[textbox, corpus],
        api_name="draw_btn_anon"
    )

    textbox.submit(
        check_input_retrieval,
        inputs=textbox,
        outputs=None,
    ).success(        
        partial(disable_buttons_side_by_side, 4),
        inputs=None,
        outputs=[textbox, corpus, send_btn, draw_btn],
    ).then(        
        gen_func,
        inputs=[state0, state1, textbox, corpus, model_selector_left, model_selector_right],
        outputs=[state0, state1, chatbot_left, chatbot_right],
        api_name="textbox_side_by_side"
    ).then(
        enable_btns, 
        inputs=None,  
        outputs=btn_list 
    )
    
    send_btn.click(
        check_input_retrieval,
        inputs=textbox,
        outputs=None,
    ).success(
        partial(disable_buttons_side_by_side, 4),
        inputs=None,
        outputs=[textbox, corpus, send_btn, draw_btn],
    ).then(
        gen_func,
        inputs=[state0, state1, textbox, corpus, model_selector_left, model_selector_right],
        outputs=[state0, state1, chatbot_left, chatbot_right],
        api_name="send_side_by_side"
    ).then(
        enable_btns,
        inputs=None,
        outputs=btn_list
    )
    
    clear_btn.click(
        clear_history_side_by_side, 
        inputs=[state0, state1], 
        outputs=[state0, state1, textbox, chatbot_left, chatbot_right, download_a_btn, download_b_btn], 
        api_name="clear_btn_side_by_side"
    ).then(
        disable_buttons_side_by_side,
        inputs=None,
        outputs=btn_list
    ).then(
        partial(enable_btns, 4),
        inputs=None,
        outputs=[textbox, corpus, send_btn, draw_btn],
    )

    leftvote_btn.click(
        partial(vote_last_response, "leftvote"),
        inputs=[state0, state1, model_selector_left, model_selector_right],
        outputs=[textbox, leftvote_btn, rightvote_btn, tie_btn, bothbad_btn, model_selector_left, model_selector_right, download_a_btn, download_b_btn]
    )
    rightvote_btn.click(
        partial(vote_last_response, "rightvote"),
        inputs=[state0, state1, model_selector_left, model_selector_right],
        outputs=[textbox, leftvote_btn, rightvote_btn, tie_btn, bothbad_btn, model_selector_left, model_selector_right, download_a_btn, download_b_btn]
    )
    tie_btn.click(
        partial(vote_last_response, "tievote"),
        inputs=[state0, state1, model_selector_left, model_selector_right],
        outputs=[textbox, leftvote_btn, rightvote_btn, tie_btn, bothbad_btn, model_selector_left, model_selector_right, download_a_btn, download_b_btn]
    )
    bothbad_btn.click(
        partial(vote_last_response, "bothbadvote"),
        inputs=[state0, state1, model_selector_left, model_selector_right],
        outputs=[textbox, leftvote_btn, rightvote_btn, tie_btn, bothbad_btn, model_selector_left, model_selector_right, download_a_btn, download_b_btn]
    )    

    share_js = """
function (a, b, c, d) {
    const captureElement = document.querySelector('#share-region-named');
    html2canvas(captureElement)
        .then(canvas => {
            canvas.style.display = 'none'
            document.body.appendChild(canvas)
            return canvas
        })
        .then(canvas => {
            const image = canvas.toDataURL('image/png')
            const a = document.createElement('a')
            a.setAttribute('download', 'mteb-arena.png')
            a.setAttribute('href', image)
            a.click()
            canvas.remove()
        });
    return [a, b, c, d];
}
"""
    share_btn.click(
        partial(vote_last_response, "share"),
        inputs=[state0, state1, model_selector_left, model_selector_right],
        outputs=[],
        js=share_js
    )

def build_single_model_ui(models):
    notice_markdown = f"""
# 💧 MTEB Arena Single Model: Retrieval 🔎
"""
    #| [GitHub](https://github.com/embeddings-benchmark) |
    ### 🤖 Choose any retriever
    state = gr.State()
    gen_func = partial(retrieve, models.retrieve)
    gr.Markdown(notice_markdown, elem_id="notice_markdown")

    model_list = models.models_retrieval

    with gr.Row(elem_id="model_selector_row"):
        model_selector = gr.Dropdown(
            choices=model_list,
            value=DEFAULT_MODEL_A,
            interactive=True,
            show_label=False
        )

    with gr.Row():
        with gr.Accordion(
            "🔍 Expand to see all model descriptions",
            open=False,
            elem_id="model_description_accordion",
        ):
            model_description_md = models.get_model_description_md()
            gr.Markdown(model_description_md, elem_id="model_description_markdown")

    with gr.Row():
        chatbot = gr.Chatbot(
            label="Model",
            elem_id="chatbot",
            height=550,
            show_copy_button=True,
        )

    with gr.Row():
        textbox = gr.Textbox(
            label="Query",
            show_label=True,
            placeholder="👉 Enter text and press ENTER",
            container=True,
            elem_id="input_box",
        )
        corpus = gr.Dropdown(
            label="Corpus",
            choices=["wikipedia", "arxiv"],#, "stackexchange"],
            value="wikipedia",
            interactive=True,
            show_label=True,
            container=True,
            scale=0,
        )
        send_btn = gr.Button(value="Send", variant="primary", scale=0)
        draw_btn = gr.Button(value="🎲 Random sample", variant="primary", scale=0)

    with gr.Row() as button_row:
        upvote_btn = gr.Button(value="👍  Upvote", interactive=False)
        downvote_btn = gr.Button(value="👎  Downvote", interactive=False)
        flag_btn = gr.Button(value="⚠️  Flag", interactive=False)
        clear_btn = gr.Button(value="🗑️  Clear history", interactive=False)

    gr.Examples(
        examples=[
            ["Which test was devised to determine whether robots can think?", "wikipedia"],
            ["I am looking for a paper to help me deal with data constraints as I scale up my language model.", "arxiv"],
#            ["What's the purpose of `if __name__ == '__main__'`?", "stackexchange"],
            ["Find me a planet near α 23h 06m 29.368s and δ −05° 02′ 29.04″ that could be habitable.", "wikipedia"],
            ["I am searching for a good and large-scale benchmark for testing the performance of text embeddings.", "arxiv"],
#            ["In English when to use –, — vs -?", "stackexchange"],
            ["Who wrote about a guide more popular than the Encyclopedia Galactica?", "wikipedia"],
            ["Please find me the paper on training code large language models to follow instructions via git commits.", "arxiv"],
#            ["Why did the machines in The Matrix keep humans around?", "stackexchange"],
        ],
        inputs=[textbox, corpus],
    )

    btn_list = [upvote_btn, downvote_btn, flag_btn, clear_btn]

    draw_btn.click(
        partial(models.retrieve_draw, ["wikipedia", "arxiv"]),
        inputs=None,
        outputs=[textbox, corpus],
        api_name="draw_btn_single"
    )

    model_selector.change(
        clear_history, 
        inputs=None,
        outputs=[state, textbox, chatbot], 
        api_name="model_selector_single"
    ).then(
        partial(disable_btns, 4),
        inputs=None,
        outputs=btn_list
    ).then(
        partial(enable_btns, 4),
        inputs=None,
        outputs=[textbox, corpus, send_btn, draw_btn],
    )

    textbox.submit(
        check_input_retrieval,
        inputs=textbox,
        outputs=None,
    ).success(
        partial(disable_buttons_side_by_side, 4),
        inputs=None,
        outputs=[textbox, corpus, send_btn, draw_btn],
    ).then(
        gen_func,
        inputs=[state, textbox, corpus, model_selector],
        outputs=[state, chatbot],
        api_name="submit_btn_single",
        show_progress = "full"
    ).success(
        partial(enable_btns, 4),
        inputs=None,
        outputs=btn_list
    )

    send_btn.click(
        check_input_retrieval,
        inputs=textbox,
        outputs=None,
    ).success(
        partial(disable_buttons_side_by_side, 4),
        inputs=None,
        outputs=[textbox, corpus, send_btn, draw_btn],
    ).then(
        gen_func,
        inputs=[state, textbox, corpus, model_selector],
        outputs=[state, chatbot],
        api_name="send_btn_single",
        show_progress = "full"
    ).success(
        partial(enable_btns, 4),
        inputs=None,
        outputs=btn_list
    )
    upvote_btn.click(
        partial(vote_last_response_single, "upvote"),
        inputs=[state, model_selector],
        outputs=[textbox, upvote_btn, downvote_btn, flag_btn]
    )
    downvote_btn.click(
        partial(vote_last_response_single, "downvote"),
        inputs=[state, model_selector],
        outputs=[textbox, upvote_btn, downvote_btn, flag_btn]
    )
    flag_btn.click(
        partial(vote_last_response_single, "flag"),
        inputs=[state, model_selector],
        outputs=[textbox, upvote_btn, downvote_btn, flag_btn]
    )
    clear_btn.click(
        clear_history,
        inputs=None,
        outputs=[state, textbox, chatbot],
        api_name="clear_history_single",
        show_progress="full"
    ).then(
        partial(disable_btns, 4),
        inputs=None,
        outputs=btn_list
    ).then(
        partial(enable_btns, 4),
        inputs=None,
        outputs=[textbox, corpus, send_btn, draw_btn],
    )

### Clustering ###

def check_input_clustering(state, txt):
    if not(txt): raise gr.Warning("Text cannot be empty.")
    if (state) and (hasattr(state, "prompts")) and (txt in state.prompts): raise gr.Warning("Text already added.")    

# https://github.com/lm-sys/FastChat/blob/73936244535664c7e4c9bc1a419aa7f77b2da88e/fastchat/serve/gradio_web_server.py#L100
# https://github.com/lm-sys/FastChat/blob/73936244535664c7e4c9bc1a419aa7f77b2da88e/fastchat/serve/gradio_block_arena_named.py#L165
class ClusteringState:
    def __init__(self, model_name):
        self.conv_id = uuid.uuid4().hex
        self.model_name = model_name
        self.prompts = []
        self.output = ""
        self.ncluster = 1
        self.ndim = "3D"
        self.dim_method = "PCA"
        self.clustering_method = "KMeans"

    def dict(self, prefix: str = None):
        if prefix is None:
            return {"conv_id": self.conv_id, "model_name": self.model_name, "prompt": self.prompts, "ncluster": self.ncluster, "output": self.output, "ndim": self.ndim, "dim_method": self.dim_method, "clustering_method": self.clustering_method}
        else:
            return {f"{prefix}_conv_id": self.conv_id, f"{prefix}_model_name": self.model_name, f"{prefix}_prompt": self.prompts, f"{prefix}_ncluster": self.ncluster, f"{prefix}_output": self.output, f"{prefix}_ndim": self.ndim, f"{prefix}_dim_method": self.dim_method, f"{prefix}_clustering_method": self.clustering_method}

def clustering_side_by_side(gen_func, state0, state1, txt, ncluster, ndim, dim_method, clustering_method, model_name0, model_name1, request: gr.Request):
    if state0 is None:
        state0 = ClusteringState(model_name1)
    if state1 is None:
        state1 = ClusteringState(model_name0)
    # txt may be None if only changing the dim
    if txt:
        if "<|SEP|>" in txt:
            state0.prompts.extend(txt.split("<|SEP|>"))
            state1.prompts.extend(txt.split("<|SEP|>"))
        else:    
            state0.prompts.append(txt)
            state1.prompts.append(txt)

    state0.ncluster = ncluster
    state1.ncluster = ncluster

    state0.ndim = ndim
    state1.ndim = ndim

    state0.dim_method = dim_method
    state1.dim_method = dim_method

    state0.clustering_method = clustering_method
    state1.clustering_method = clustering_method

    ip = get_ip(request)
    clustering_logger.info(f"Clustering. ip: {ip}")
    start_tstamp = time.time()
    # Remove prefixes in case it is already beyoned the 1st round.
    model_name0, model_name1 = model_name0.replace("### Model A: ", ""), model_name1.replace("### Model B: ", "")
    generated_image0, generated_image1, model_name0, model_name1 = gen_func(state0.prompts, model_name0, model_name1, ncluster, ndim=ndim.split(" ")[0], dim_method=dim_method, clustering_method=clustering_method)
    state0.model_name, state1.model_name = model_name0, model_name1
    
    yield state0, state1, generated_image0, generated_image1, None, \
        gr.Markdown(f"### Model A: {model_name0}", visible=False), gr.Markdown(f"### Model B: {model_name1}", visible=False)
    
    finish_tstamp = time.time()
    
    data = {
        "tstamp": round(finish_tstamp, 4),
        "task_type": "clustering",
        "type": "chat",
        "model": model_name0,
        "gen_params": {},
        "start": round(start_tstamp, 4),
        "finish": round(finish_tstamp, 4),
        "ip": get_ip(request),
        **state0.dict()
    }
    store_data_in_hub(data, "clustering_individual")
    
    data = {
        "tstamp": round(finish_tstamp, 4),
        "task_type": "clustering",
        "type": "chat",
        "model": model_name1,
        "gen_params": {},
        "start": round(start_tstamp, 4),
        "finish": round(finish_tstamp, 4),
        "ip": get_ip(request),
        **state1.dict()
    }
    store_data_in_hub(data, "clustering_individual")


def clustering(gen_func, state, txt, ncluster, ndim, dim_method, clustering_method, model_name, request: gr.Request):
    if not model_name: raise gr.Warning("Model name cannot be empty.")
    if state is None:
        state = ClusteringState(model_name)
    ip = get_ip(request)
    clustering_logger.info(f"Clustering. ip: {ip}")
    start_tstamp = time.time()
    # txt may be None if only changing the dim
    if txt:
        if "<|SEP|>" in txt:
            state.prompts.extend(txt.split("<|SEP|>"))
        else:
            state.prompts.append(txt)
    state.ncluster = ncluster
    generated_img = gen_func(state.prompts, model_name, state.ncluster, ndim=ndim.split(" ")[0], dim_method=dim_method, clustering_method=clustering_method)
    state.model_name = model_name

    yield state, generated_img, None
    
    finish_tstamp = time.time()

    data = {
        "tstamp": round(finish_tstamp, 4),
        "task_type": "clustering",
        "type": "chat",
        "model": model_name,
        "gen_params": {},
        "start": round(start_tstamp, 4),
        "finish": round(finish_tstamp, 4),
        "ip": get_ip(request),
        **state.dict()
    }
    store_data_in_hub(data, "clustering_individual")

def toggle_btn(btn):
    if btn == "3D (press for 2D)": return gr.update(value="2D (press for 3D)", variant="primary")
    else: return gr.update(value="3D (press for 2D)", variant="primary")
    
def check_input_clustering_dim(state):
    if not(state) or not(state.prompts): raise

def build_side_by_side_ui_anon_clustering(models):
    notice_markdown = """
# ⚔️ MTEB Arena: Clustering ✨

## 📜 Rules (Play the video for an explanation ➡️)
- Input & submit texts one-by-one to two anonymous models & vote which clusters them better.
- You can enter >1 texts at once if you separate them with `<|SEP|>` like in the examples.
- If you specify a number of clusters >1, a KMeans will be trained on the embeddings and clusters colored according to its predictions.
- Clusters are 1D for 1 text, 2D for 2-3 texts, 3D for >3 texts.
- You have to **enter at least 3 texts**, else cluster qualities cannot be judged.

## 👇 Vote now!
"""
    state0 = gr.State(None)
    state1 = gr.State(None)
    gen_func = partial(clustering_side_by_side, models.clustering_parallel)    

    with gr.Row():
        with gr.Column():
            gr.Markdown(notice_markdown, elem_id="notice_markdown")
        with gr.Column():
            gr.Video("videos/clustering_explanation.mp4", label="Video Explanation", elem_id="video")

    with gr.Group(elem_id="share-region-anon"):
        with gr.Accordion("🔍 Expand to see all Arena players", open=False):
            model_description_md = models.get_model_description_md(task_type="clustering")
            gr.Markdown(model_description_md, elem_id="model_description_markdown")
        with gr.Row():
            with gr.Column():
                chatbot_left = gr.Plot(label="Model A")
            with gr.Column():
                chatbot_right = gr.Plot(label="Model B")

        with gr.Row():
            with gr.Column():
                model_selector_left = gr.Markdown("", visible=False)
            with gr.Column():
                model_selector_right = gr.Markdown("", visible=False)

        with gr.Row():
            leftvote_btn = gr.Button(
                value="👈  A is better", visible=False, interactive=False
            )
            rightvote_btn = gr.Button(
                value="👉  B is better", visible=False, interactive=False
            )
            tie_btn = gr.Button(value="🤝  Tie", visible=False, interactive=False)
            bothbad_btn = gr.Button(
                value="👎  Both are bad", visible=False, interactive=False
            )

    with gr.Row():
        textbox = gr.Textbox(
            show_label=True,
            label="Text to cluster",
            placeholder="👉 Enter your text and press ENTER",
            elem_id="input_box",
            scale=64,
        )
        ncluster = gr.Number(
            show_label=True,
            label="Number of clusters",
            elem_id="ncluster_box",
            value=1,
            minimum=1,
            scale=12,
            min_width=0,
        )
        send_btn = gr.Button(value="Send", variant="primary", scale=8, min_width=0)
        draw_btn = gr.Button(value="🎲 Random sample", variant="primary", scale=8, min_width=0)
        dim_btn = gr.Button(value="3D (press for 2D)", variant="primary", scale=5, min_width=0)

    with gr.Row():
        clear_btn = gr.Button(value="🎲 New Round", interactive=False)

    with gr.Accordion("⚙️ Parameters", open=False) as parameter_row:
        with gr.Row():
            dim_method = gr.Radio(
                ["PCA", "UMAP", "TSNE"],
                value="PCA",
                interactive=True,
                label="Dimensioality reduction algorithm",
            )
            clustering_method = gr.Radio(
                ["KMeans", "MiniBatchKMeans"],
                value="KMeans",
                interactive=True,
                label="Clustering algorithm",
            )
        
    gr.Examples(
        examples=[
            ["Shanghai<|SEP|>Beijing<|SEP|>Shenzhen<|SEP|>Hangzhou<|SEP|>Seattle<|SEP|>Boston<|SEP|>New York<|SEP|>San Francisco", 2],
            ["Pikachu<|SEP|>Darth Vader<|SEP|>Yoda<|SEP|>Squirtle<|SEP|>Gandalf<|SEP|>Legolas<|SEP|>Mickey Mouse<|SEP|>Donald Duck<|SEP|>Charizard", 4],
            # https://www.reddit.com/r/Bitcoin/top/?t=all ; https://www.reddit.com/r/longevity/top/?t=all ; https://www.reddit.com/r/MachineLearning/top/?t=all
            ["It's official! 1 Bitcoin = $10,000 USD<|SEP|>Everyone who's trading BTC right now<|SEP|>Age reversal not only achievable but also possibly imminent: Retro Biosciences<|SEP|>MicroRNA regrows 90% of lost hair, study finds<|SEP|>Speech-to-speech translation for a real-world unwritten language<|SEP|>Seeking the Best Embedding Model: Experiences with the MTEB Arena?", 3],
            ["If someone online buys something off of my Amazon wish list, do they get my full name and address?<|SEP|>Package \"In Transit\" over a week. No scheduled delivery date, no locations. What's up?<|SEP|>Can Amazon gift cards replace a debit card?<|SEP|>Homesick GWS star Cameron McCarthy on road to recovery<|SEP|>Accidently ordered 2 of an item, how do I only return 1? For free?<|SEP|>Need help ASAP, someone ordering in my account<|SEP|>So who's everyone tipping for Round 1?", 2],
            ["octagon<|SEP|>rectangle<|SEP|>Temple of Artemis<|SEP|>Colossus of Rhodes<|SEP|>Statue of Zeus<|SEP|>Lighthouse of Alexandria<|SEP|>Hanging Gardens of Babylon<|SEP|>Pyramids of Giza<|SEP|>brunette<|SEP|>black<|SEP|>blonde<|SEP|>redhead<|SEP|>gray<|SEP|>auburn<|SEP|>white<|SEP|>soccer<|SEP|>basketball<|SEP|>tennis<|SEP|>baseball<|SEP|>cricket<|SEP|>ruby<|SEP|>topaz<|SEP|>diamond", 5],
        ],
        inputs=[textbox, ncluster],
    )

    btn_list = [leftvote_btn, rightvote_btn, tie_btn, bothbad_btn, clear_btn]

    dim_btn.click(
        toggle_btn,
        inputs=[dim_btn],
        outputs=[dim_btn],
    ).then(
        check_input_clustering_dim,
        inputs=[state0],
        outputs=None,
    ).success(
        partial(disable_btns, 5, visible=False),
        inputs=None,
        outputs=[textbox, ncluster, send_btn, draw_btn, dim_btn],
    ).then(
        gen_func,
        inputs=[state0, state1, textbox, ncluster, dim_btn, dim_method, clustering_method, model_selector_left, model_selector_right],
        outputs=[state0, state1, chatbot_left, chatbot_right, textbox, model_selector_left, model_selector_right],
        api_name="textbox_side_by_side"
    ).then(
        enable_btns_clustering,
        inputs=state0,
        outputs=[textbox, ncluster, send_btn, draw_btn, dim_btn] + btn_list,
    )

    draw_btn.click(
        models.clustering_draw,
        inputs=None,
        outputs=[textbox, ncluster],
        api_name="draw_btn_anon"
    )

    textbox.submit(
        check_input_clustering,
        inputs=[state0, textbox],
        outputs=None,
    ).success(
        partial(disable_btns, 5, visible=False),
        inputs=None,
        outputs=[textbox, ncluster, send_btn, draw_btn, dim_btn],
    ).then(
        gen_func,
        inputs=[state0, state1, textbox, ncluster, dim_btn, dim_method, clustering_method, model_selector_left, model_selector_right],
        outputs=[state0, state1, chatbot_left, chatbot_right, textbox, model_selector_left, model_selector_right],
        api_name="submit_btn_anon"
    ).then(
        enable_btns_clustering,
        inputs=state0,
        outputs=[textbox, ncluster, send_btn, draw_btn, dim_btn] + btn_list,
    )

    send_btn.click(
        check_input_clustering,
        inputs=[state0, textbox],
        outputs=None,
    ).success(        
        partial(disable_btns, 5, visible=False),
        inputs=None,
        outputs=[textbox, ncluster, send_btn, draw_btn, dim_btn],
    ).then(
        gen_func,
        inputs=[state0, state1, textbox, ncluster, dim_btn, dim_method, clustering_method, model_selector_left, model_selector_right],
        outputs=[state0, state1, chatbot_left, chatbot_right, textbox, model_selector_left, model_selector_right],
        api_name="send_btn_anon"
    ).then(
        enable_btns_clustering,
        inputs=state0,
        outputs=[textbox, ncluster, send_btn, draw_btn, dim_btn] + btn_list,
    )

    clear_btn.click(
        clear_history_side_by_side_anon_clustering,
        inputs=None,
        outputs=[state0, state1, textbox, ncluster, chatbot_left, chatbot_right, model_selector_left, model_selector_right],
        api_name="clear_btn_anon"
    ).then(
        disable_buttons_side_by_side,
        inputs=None,
        outputs=btn_list
    ).then(
        partial(enable_btns, 4),
        inputs=None,
        outputs=[send_btn, textbox, ncluster, draw_btn],
    )

    dummy_left_model = gr.State("")
    dummy_right_model = gr.State("")
    leftvote_btn.click(
        partial(vote_last_response_clustering, "leftvote"),
        inputs=[state0, state1, dummy_left_model, dummy_right_model],
        outputs=[send_btn, draw_btn, dim_btn, textbox, ncluster, leftvote_btn, rightvote_btn, tie_btn, bothbad_btn, model_selector_left, model_selector_right]
    )
    rightvote_btn.click(
        partial(vote_last_response_clustering, "rightvote"),
        inputs=[state0, state1, dummy_left_model, dummy_right_model],
        outputs=[send_btn, draw_btn, dim_btn, textbox, ncluster, leftvote_btn, rightvote_btn, tie_btn, bothbad_btn, model_selector_left, model_selector_right]
    )
    tie_btn.click(
        partial(vote_last_response_clustering, "tievote"),
        inputs=[state0, state1, dummy_left_model, dummy_right_model],
        outputs=[send_btn, draw_btn, dim_btn, textbox, ncluster, leftvote_btn, rightvote_btn, tie_btn, bothbad_btn, model_selector_left, model_selector_right]
    )
    bothbad_btn.click(
        partial(vote_last_response_clustering, "bothbadvote"),
        inputs=[state0, state1, dummy_left_model, dummy_right_model],
        outputs=[send_btn, draw_btn, dim_btn, textbox, ncluster, leftvote_btn, rightvote_btn, tie_btn, bothbad_btn, model_selector_left, model_selector_right]
    )

def build_side_by_side_ui_named_clustering(models):
    notice_markdown = """
# ⚔️ MTEB Arena: Clustering ✨

## 📜 Rules (Play the video for an explanation ➡️)
- Input & submit texts one-by-one to two models & vote which clusters them better.
- You can enter >1 texts at once if you separate them with `<|SEP|>` like in the examples.
- If you specify a number of clusters >1, a KMeans will be trained on the embeddings and clusters colored according to its predictions.
- Clusters are 1D for 1 text, 2D for 2-3 texts, 3D for >3 texts.
- You have to **enter at least 3 texts**, else cluster qualities cannot be judged.

## 👇 Vote now!
"""
    model_list = models.models_clustering

    state0 = gr.State(None)
    state1 = gr.State(None)
    gen_func = partial(clustering_side_by_side, models.clustering_parallel)    

    with gr.Row():
        with gr.Column():
            gr.Markdown(notice_markdown, elem_id="notice_markdown")
        with gr.Column():
            gr.Video("videos/clustering_explanation.mp4", label="Video Explanation", elem_id="video")

    with gr.Group(elem_id="share-region-named"):
        with gr.Row():
            with gr.Column():
                model_selector_left = gr.Dropdown(
                    choices=model_list,
                    value=DEFAULT_MODEL_A,
                    interactive=True,
                    show_label=False,
                    container=False,
                )
            with gr.Column():
                model_selector_right = gr.Dropdown(
                    choices=model_list,
                    value=DEFAULT_MODEL_B,
                    interactive=True,
                    show_label=False,
                    container=False,
                )
        with gr.Row():
            with gr.Accordion("🔍 Expand to see all model descriptions", open=False):
                model_description_md = models.get_model_description_md(task_type="clustering")
                gr.Markdown(model_description_md, elem_id="model_description_markdown")

        with gr.Row():
            with gr.Column():
                chatbot_left = gr.Plot(label="Model A")
            with gr.Column():
                chatbot_right = gr.Plot(label="Model B")
        with gr.Row():
            leftvote_btn = gr.Button(
                value="👈  A is better", visible=False, interactive=False
            )
            rightvote_btn = gr.Button(
                value="👉  B is better", visible=False, interactive=False
            )
            tie_btn = gr.Button(value="🤝  Tie", visible=False, interactive=False)
            bothbad_btn = gr.Button(
                value="👎  Both are bad", visible=False, interactive=False
            )

    with gr.Row():
        textbox = gr.Textbox(
            show_label=True,
            label="Text to cluster",
            placeholder="👉 Enter your text and press ENTER",
            elem_id="input_box",
            scale=64,
        )
        ncluster = gr.Number(
            show_label=True,
            label="Number of clusters",
            elem_id="ncluster_box",
            value=1,
            minimum=1,
            scale=12,
            min_width=0,
        )
        send_btn = gr.Button(value="Send", variant="primary", scale=8, min_width=0)
        draw_btn = gr.Button(value="🎲 Random sample", variant="primary", scale=8, min_width=0)
        dim_btn = gr.Button(value="3D (press for 2D)", variant="primary", scale=5, min_width=0)

    with gr.Row():
        clear_btn = gr.Button(value="🗑️  Clear history", interactive=False)

    with gr.Accordion("⚙️ Parameters", open=False) as parameter_row:
        with gr.Row():
            dim_method = gr.Radio(
                ["PCA", "UMAP", "TSNE"],
                value="PCA",
                interactive=True,
                label="Dimensioality reduction algorithm",
            )
            clustering_method = gr.Radio(
                ["KMeans", "MiniBatchKMeans"],
                value="KMeans",
                interactive=True,
                label="Clustering algorithm",
            )

    gr.Examples(
        examples=[
            ["Shanghai<|SEP|>Beijing<|SEP|>Shenzhen<|SEP|>Hangzhou<|SEP|>Seattle<|SEP|>Boston<|SEP|>New York<|SEP|>San Francisco", 2],
            ["Pikachu<|SEP|>Darth Vader<|SEP|>Yoda<|SEP|>Squirtle<|SEP|>Gandalf<|SEP|>Legolas<|SEP|>Mickey Mouse<|SEP|>Donald Duck<|SEP|>Charizard", 4],
            # https://www.reddit.com/r/Bitcoin/top/?t=all ; https://www.reddit.com/r/longevity/top/?t=all ; https://www.reddit.com/r/MachineLearning/top/?t=all
            ["It's official! 1 Bitcoin = $10,000 USD<|SEP|>Everyone who's trading BTC right now<|SEP|>Age reversal not only achievable but also possibly imminent: Retro Biosciences<|SEP|>MicroRNA regrows 90% of lost hair, study finds<|SEP|>Speech-to-speech translation for a real-world unwritten language<|SEP|>Seeking the Best Embedding Model: Experiences with the MTEB Arena?", 3],
            ["If someone online buys something off of my Amazon wish list, do they get my full name and address?<|SEP|>Package \"In Transit\" over a week. No scheduled delivery date, no locations. What's up?<|SEP|>Can Amazon gift cards replace a debit card?<|SEP|>Homesick GWS star Cameron McCarthy on road to recovery<|SEP|>Accidently ordered 2 of an item, how do I only return 1? For free?<|SEP|>Need help ASAP, someone ordering in my account<|SEP|>So who's everyone tipping for Round 1?", 2],
            ["octagon<|SEP|>rectangle<|SEP|>Temple of Artemis<|SEP|>Colossus of Rhodes<|SEP|>Statue of Zeus<|SEP|>Lighthouse of Alexandria<|SEP|>Hanging Gardens of Babylon<|SEP|>Pyramids of Giza<|SEP|>brunette<|SEP|>black<|SEP|>blonde<|SEP|>redhead<|SEP|>gray<|SEP|>auburn<|SEP|>white<|SEP|>soccer<|SEP|>basketball<|SEP|>tennis<|SEP|>baseball<|SEP|>cricket<|SEP|>ruby<|SEP|>topaz<|SEP|>diamond", 5],
        ],
        inputs=[textbox, ncluster],
    )

    btn_list = [leftvote_btn, rightvote_btn, tie_btn, bothbad_btn, clear_btn]

    dim_btn.click(
        toggle_btn,
        inputs=[dim_btn],
        outputs=[dim_btn],
    ).then(
        check_input_clustering_dim,
        inputs=[state0],
        outputs=None,
    ).success(
        partial(disable_btns, 5, visible=False),
        inputs=None,
        outputs=[textbox, ncluster, send_btn, draw_btn, dim_btn],
    ).then(
        gen_func,
        inputs=[state0, state1, textbox, ncluster, dim_btn, dim_method, clustering_method, model_selector_left, model_selector_right],
        outputs=[state0, state1, chatbot_left, chatbot_right, textbox],
        api_name="textbox_side_by_side"
    ).then(
        enable_btns_clustering,
        inputs=state0,
        outputs=[textbox, ncluster, send_btn, draw_btn, dim_btn] + btn_list,
    )

    draw_btn.click(
        models.clustering_draw,
        inputs=None,
        outputs=[textbox, ncluster],
        api_name="draw_btn_anon"
    )

    textbox.submit(
        check_input_clustering,
        inputs=[state0, textbox],
        outputs=None,
    ).success(
        partial(disable_btns, 5, visible=False),
        inputs=None,
        outputs=[send_btn, textbox, ncluster, draw_btn, dim_btn],
    ).then(
        gen_func,
        inputs=[state0, state1, textbox, ncluster, dim_btn, dim_method, clustering_method, model_selector_left, model_selector_right],
        outputs=[state0, state1, chatbot_left, chatbot_right, textbox],
        api_name="textbox_side_by_side"
    ).then(
        enable_btns_clustering,
        inputs=state0,
        outputs=[textbox, ncluster, send_btn, draw_btn, dim_btn] + btn_list,
    )

    send_btn.click(
        check_input_clustering,
        inputs=[state0, textbox],
        outputs=None,
    ).success(        
        partial(disable_btns, 5, visible=False),
        inputs=None,
        outputs=[send_btn, draw_btn, textbox, ncluster, dim_btn],
    ).then(
        gen_func,
        inputs=[state0, state1, textbox, ncluster, dim_btn, dim_method, clustering_method, model_selector_left, model_selector_right],
        outputs=[state0, state1, chatbot_left, chatbot_right, textbox],
        api_name="send_side_by_side"
    ).then(
        enable_btns_clustering,
        inputs=state0,
        outputs=[textbox, ncluster, send_btn, draw_btn, dim_btn] + btn_list,
    )

    clear_btn.click(
        clear_history_side_by_side_anon_clustering,
        inputs=None,
        outputs=[state0, state1, textbox, ncluster, chatbot_left, chatbot_right],
        api_name="clear_btn_anon"
    ).then(
        disable_buttons_side_by_side,
        inputs=None,
        outputs=btn_list
    ).then(
        partial(enable_btns, 4),
        inputs=None,
        outputs=[send_btn, textbox, ncluster, draw_btn],
    )

    leftvote_btn.click(
        partial(vote_last_response_clustering, "leftvote"),
        inputs=[state0, state1, model_selector_left, model_selector_right],
        outputs=[send_btn, draw_btn, dim_btn, textbox, ncluster, leftvote_btn, rightvote_btn, tie_btn, bothbad_btn, model_selector_left, model_selector_right]
    )
    rightvote_btn.click(
        partial(vote_last_response_clustering, "rightvote"),
        inputs=[state0, state1, model_selector_left, model_selector_right],
        outputs=[send_btn, draw_btn, dim_btn, textbox, ncluster, leftvote_btn, rightvote_btn, tie_btn, bothbad_btn, model_selector_left, model_selector_right]
    )
    tie_btn.click(
        partial(vote_last_response_clustering, "tievote"),
        inputs=[state0, state1, model_selector_left, model_selector_right],
        outputs=[send_btn, draw_btn, dim_btn, textbox, ncluster, leftvote_btn, rightvote_btn, tie_btn, bothbad_btn, model_selector_left, model_selector_right]
    )
    bothbad_btn.click(
        partial(vote_last_response_clustering, "bothbadvote"),
        inputs=[state0, state1, model_selector_left, model_selector_right],
        outputs=[send_btn, draw_btn, dim_btn, textbox, ncluster, leftvote_btn, rightvote_btn, tie_btn, bothbad_btn, model_selector_left, model_selector_right]
    )

def build_single_model_ui_clustering(models):
    notice_markdown = f"""
# 💧 MTEB Arena Single Model: Clustering ✨
"""
    # | [GitHub](https://github.com/embeddings-benchmark) | 
    ## 🤖 Choose any clustering model

    state = gr.State(None)
    gen_func = partial(clustering, models.clustering)
    gr.Markdown(notice_markdown, elem_id="notice_markdown")

    model_list = models.models_clustering

    with gr.Row(elem_id="model_selector_row"):
        model_selector = gr.Dropdown(
            choices=model_list,
            value=DEFAULT_MODEL_A,
            interactive=True,
            show_label=False
        )

    with gr.Row():
        with gr.Accordion(
            "🔍 Expand to see all model descriptions",
            open=False,
            elem_id="model_description_accordion",
        ):
            model_description_md = models.get_model_description_md(task_type="clustering")
            gr.Markdown(model_description_md, elem_id="model_description_markdown")

    with gr.Row():
        chatbot = gr.Plot(label="Model")

    with gr.Row():
        textbox = gr.Textbox(
            show_label=True,
            label="Text to cluster",
            placeholder="👉 Enter your text and press ENTER",
            elem_id="input_box",
            scale=64,
        )
        ncluster = gr.Number(
            show_label=True,
            label="Number of clusters",
            elem_id="ncluster_box",
            value=1,
            minimum=1,
            scale=12,
            min_width=0,
        )
        send_btn = gr.Button(value="Send", variant="primary", scale=8, min_width=0)
        draw_btn = gr.Button(value="🎲 Random sample", variant="primary", scale=8, min_width=0)
        dim_btn = gr.Button(value="3D (press for 2D)", variant="primary", scale=5, min_width=0)

    with gr.Row() as button_row:
        upvote_btn = gr.Button(value="👍  Upvote", interactive=False)
        downvote_btn = gr.Button(value="👎  Downvote", interactive=False)
        flag_btn = gr.Button(value="⚠️  Flag", interactive=False)
        clear_btn = gr.Button(value="🗑️  Clear history", interactive=False)

    with gr.Accordion("⚙️ Parameters", open=False) as parameter_row:
        with gr.Row():
            dim_method = gr.Radio(
                ["PCA", "UMAP", "TSNE"],
                value="PCA",
                interactive=True,
                label="Dimensioality reduction algorithm",
            )
            clustering_method = gr.Radio(
                ["KMeans", "MiniBatchKMeans"],
                value="KMeans",
                interactive=True,
                label="Clustering algorithm",
            )

    gr.Examples(
        examples=[
            ["Shanghai<|SEP|>Beijing<|SEP|>Shenzhen<|SEP|>Hangzhou<|SEP|>Seattle<|SEP|>Boston<|SEP|>New York<|SEP|>San Francisco", 2],
            ["Pikachu<|SEP|>Darth Vader<|SEP|>Yoda<|SEP|>Squirtle<|SEP|>Gandalf<|SEP|>Legolas<|SEP|>Mickey Mouse<|SEP|>Donald Duck<|SEP|>Charizard", 4],
            # https://www.reddit.com/r/Bitcoin/top/?t=all ; https://www.reddit.com/r/longevity/top/?t=all ; https://www.reddit.com/r/MachineLearning/top/?t=all
            ["It's official! 1 Bitcoin = $10,000 USD<|SEP|>Everyone who's trading BTC right now<|SEP|>Age reversal not only achievable but also possibly imminent: Retro Biosciences<|SEP|>MicroRNA regrows 90% of lost hair, study finds<|SEP|>Speech-to-speech translation for a real-world unwritten language<|SEP|>Seeking the Best Embedding Model: Experiences with the MTEB Arena?", 3],
            ["If someone online buys something off of my Amazon wish list, do they get my full name and address?<|SEP|>Package \"In Transit\" over a week. No scheduled delivery date, no locations. What's up?<|SEP|>Can Amazon gift cards replace a debit card?<|SEP|>Homesick GWS star Cameron McCarthy on road to recovery<|SEP|>Accidently ordered 2 of an item, how do I only return 1? For free?<|SEP|>Need help ASAP, someone ordering in my account<|SEP|>So who's everyone tipping for Round 1?", 2],
            ["octagon<|SEP|>rectangle<|SEP|>Temple of Artemis<|SEP|>Colossus of Rhodes<|SEP|>Statue of Zeus<|SEP|>Lighthouse of Alexandria<|SEP|>Hanging Gardens of Babylon<|SEP|>Pyramids of Giza<|SEP|>brunette<|SEP|>black<|SEP|>blonde<|SEP|>redhead<|SEP|>gray<|SEP|>auburn<|SEP|>white<|SEP|>soccer<|SEP|>basketball<|SEP|>tennis<|SEP|>baseball<|SEP|>cricket<|SEP|>ruby<|SEP|>topaz<|SEP|>diamond", 5],
        ],
        inputs=[textbox, ncluster],
    )

    btn_list = [upvote_btn, downvote_btn, flag_btn, clear_btn]

    draw_btn.click(
        models.clustering_draw,
        inputs=None,
        outputs=[textbox, ncluster],
        api_name="draw_btn_anon"
    )

    model_selector.change(
        clear_history_clustering,
        inputs=None, 
        outputs=[state, textbox, ncluster, chatbot], 
        api_name="model_selector_single"
    ).then(
        partial(disable_btns, 4),
        inputs=None,
        outputs=btn_list
    ).then(
        partial(enable_btns, 4),
        inputs=None,
        outputs=[send_btn, draw_btn, textbox, ncluster],
    )
    
    dim_btn.click(
        toggle_btn,
        inputs=[dim_btn],
        outputs=[dim_btn],
    ).then(
        check_input_clustering_dim,
        inputs=[state],
        outputs=None,
    ).success(
        partial(disable_btns, 5, visible=False),
        inputs=None,
        outputs=[textbox, ncluster, send_btn, draw_btn, dim_btn],
    ).then(
        gen_func,
        inputs=[state, textbox, ncluster, dim_btn, dim_method, clustering_method, model_selector],
        outputs=[state, chatbot, textbox],
        api_name="submit_btn_single",
        show_progress="full"
    ).then(
        enable_btns_clustering,
        inputs=state,
        outputs=[textbox, ncluster, send_btn, draw_btn, dim_btn] + btn_list,
    )
    
    textbox.submit(
        check_input_clustering,
        inputs=[state, textbox],
        outputs=None,
    ).success(
        partial(disable_btns, 5, visible=False),
        inputs=None,
        outputs=[textbox, ncluster, send_btn, draw_btn, dim_btn],
    ).then(
        gen_func,
        inputs=[state, textbox, ncluster, dim_btn, dim_method, clustering_method, model_selector],
        outputs=[state, chatbot, textbox],
        api_name="submit_btn_single",
        show_progress="full"
    ).then(
        enable_btns_clustering,
        inputs=state,
        outputs=[textbox, ncluster, send_btn, draw_btn, dim_btn] + btn_list,
    )

    send_btn.click(
        check_input_clustering,
        inputs=[state, textbox],
        outputs=None,
    ).success(
        partial(disable_btns, 5, visible=False),
        inputs=None,
        outputs=[textbox, ncluster, send_btn, draw_btn, dim_btn],
    ).then(
        gen_func,
        inputs=[state, textbox, ncluster, dim_btn, dim_method, clustering_method, model_selector],
        outputs=[state, chatbot, textbox],
        api_name="send_btn_single",
        show_progress="full"
    ).then(
        enable_btns_clustering,
        inputs=state,
        outputs=[textbox, ncluster, send_btn, draw_btn, dim_btn] + btn_list,
    )

    upvote_btn.click(
        partial(vote_last_response_single_clustering, "upvote"),
        inputs=[state, model_selector],
        outputs=[upvote_btn, downvote_btn, flag_btn]
    )
    downvote_btn.click(
        partial(vote_last_response_single_clustering, "downvote"),
        inputs=[state, model_selector],
        outputs=[upvote_btn, downvote_btn, flag_btn]
    )
    flag_btn.click(
        partial(vote_last_response_single_clustering, "flag"),
        inputs=[state, model_selector],
        outputs=[upvote_btn, downvote_btn, flag_btn]
    )
    clear_btn.click(
        clear_history_clustering,
        inputs=None,
        outputs=[state, textbox, ncluster, chatbot],
        api_name="clear_history_single",
        show_progress="full"
    ).then(
        partial(disable_btns, 4),
        inputs=None,
        outputs=btn_list
    ).then(
        partial(enable_btns_clustering, 4),
        inputs=None,
        outputs=[send_btn, draw_btn, textbox, ncluster],
    )

### STS ###
class STSState:
    def __init__(self, model_name):
        self.conv_id = uuid.uuid4().hex
        self.model_name = model_name
        self.txt0 = ""
        self.txt1 = ""
        self.txt2 = ""
        self.output = ""

    def dict(self, prefix: str = None):
        if prefix is None:
            return {"conv_id": self.conv_id, "model_name": self.model_name, "txt0": self.txt0, "txt1": self.txt1, "txt2": self.txt2, "output": self.output}
        else:
            return {f"{prefix}_conv_id": self.conv_id, f"{prefix}_model_name": self.model_name, f"{prefix}_txt0": self.txt0, f"{prefix}_txt1": self.txt1, f"{prefix}_txt2": self.txt2, f"{prefix}_output": self.output}
        

def sts_side_by_side(gen_func, state0, state1, txt0, txt1, txt2, model_name0, model_name1, request: gr.Request):
    if any([x is None for x in (txt0, txt1, txt2)]): raise gr.Warning("Text cannot be empty.")
    state0, state1 = STSState(model_name0), STSState(model_name1)
    ip = get_ip(request)
    retrieval_logger.info(f"Retrieval. ip: {ip}")
    start_tstamp = time.time()
    generated_image0, generated_image1, model_name0, model_name1 = gen_func(txt0, txt1, txt2, model_name0, model_name1)
    state0.txt0, state0.txt1, state0.txt2 = txt0, txt1, txt2
    state1.txt0, state1.txt1, state1.txt2 = txt0, txt1, txt2
    state0.model_name, state1.model_name = model_name0, model_name1
    
    yield state0, state1, generated_image0, generated_image1, \
        gr.Markdown(f"### Model A: {model_name0}", visible=False), gr.Markdown(f"### Model B: {model_name1}", visible=False)
    
    finish_tstamp = time.time()
    
    data = {
        "tstamp": round(finish_tstamp, 4),
        "task_type": "sts",
        "type": "chat",
        "model": model_name0,
        "gen_params": {},
        "start": round(start_tstamp, 4),
        "finish": round(finish_tstamp, 4),
        "ip": get_ip(request),
        **state0.dict()
    }
    store_data_in_hub(data, "sts_individual")

    data = {
        "tstamp": round(finish_tstamp, 4),
        "task_type": "sts",
        "type": "chat",
        "model": model_name1,
        "gen_params": {},
        "start": round(start_tstamp, 4),
        "finish": round(finish_tstamp, 4),
        "ip": get_ip(request),
        **state1.dict()
    }
    store_data_in_hub(data, "sts_individual")

def sts(gen_func, state, txt0, txt1, txt2, model_name, request: gr.Request):
    if any([x is None for x in (txt0, txt1, txt2)]): raise gr.Warning("Text cannot be empty.")
    if not model_name: raise gr.Warning("Model name cannot be empty.")
    state = STSState(model_name)
    ip = get_ip(request)
    retrieval_logger.info(f"Retrieval. ip: {ip}")
    start_tstamp = time.time()
    generated_image = gen_func(txt0, txt1, txt2, model_name)
    state.txt0, state.txt1, state.txt2 = txt0, txt1, txt2
    # state.output = generated_image
    state.model_name = model_name
    
    yield state, generated_image

    finish_tstamp = time.time()
    
    data = {
        "tstamp": round(finish_tstamp, 4),
        "task_type": "sts",
        "type": "chat",
        "model": model_name,
        "gen_params": {},
        "start": round(start_tstamp, 4),
        "finish": round(finish_tstamp, 4),
        "ip": get_ip(request),
        **state.dict()
    }
    store_data_in_hub(data, "sts_individual")


def check_input_sts(txt0, txt1, txt2):
    if any([not(x) for x in (txt0, txt1, txt2)]): raise gr.Warning("Text cannot be empty.")
    if len(set([txt0, txt1, txt2])) != 3: raise gr.Warning("Please input three different texts.")

def build_side_by_side_ui_anon_sts(models):
    notice_markdown = """
# ⚔️ MTEB Arena: STS ☘️
## 📜 Rules (Play the video for an explanation ➡️)
- Input three different texts to two anonymous models and vote which visualizes their similarity better.
- The closer (smaller distance) two corners of the triangle are, the more similar their texts are to the model.
- The distances between the corners are inverted and scaled cosine similarities and displayed on the edges.

## 👇 Vote now!
"""
    state0 = gr.State()
    state1 = gr.State()
    gen_func = partial(sts_side_by_side, models.sts_parallel)

    with gr.Row():
        with gr.Column(scale=3):
            gr.Markdown(notice_markdown, elem_id="notice_markdown")
        with gr.Column(scale=2):
            gr.Video("videos/sts_explanation.mp4", label="Video Explanation", elem_id="video")

    with gr.Group(elem_id="share-region-anon"):
        with gr.Accordion("🔍 Expand to see all Arena players", open=False):
            model_description_md = models.get_model_description_md(task_type="sts")
            gr.Markdown(model_description_md, elem_id="model_description_markdown")
        with gr.Row():
            with gr.Column():
                chatbot_left = gr.Plot(label="Model A")
            with gr.Column():
                chatbot_right = gr.Plot(label="Model B")

        with gr.Row():
            with gr.Column():
                model_selector_left = gr.Markdown("", visible=False)
            with gr.Column():
                model_selector_right = gr.Markdown("", visible=False)

        with gr.Row():
            leftvote_btn = gr.Button(
                value="👈  A is better", visible=False, interactive=False
            )
            rightvote_btn = gr.Button(
                value="👉  B is better", visible=False, interactive=False
            )
            tie_btn = gr.Button(value="🤝  Tie", visible=False, interactive=False)
            bothbad_btn = gr.Button(
                value="👎  Both are bad", visible=False, interactive=False
            )

    with gr.Row():
        textbox0 = gr.Textbox(
            show_label=True,
            label="Text (1)",
            placeholder="👉 Enter text",
            container=True,
            elem_id="input_box",
        )
        textbox1 = gr.Textbox(
            show_label=True,
            label="Text (2)",
            placeholder="👉 Enter text",
            container=True,
            elem_id="input_box",
        )
        textbox2 = gr.Textbox(
            show_label=True,
            label="Text (3)",
            placeholder="👉 Enter text",
            container=True,
            elem_id="input_box",
        )
        send_btn = gr.Button(value="Send", variant="primary", scale=0)
        draw_btn = gr.Button(value="🎲 Random sample", variant="primary", scale=0)

    with gr.Row():
        clear_btn = gr.Button(value="🎲 New Round", interactive=False)

    gr.Examples(
        examples=[
            ["hello", "good morning", "早上好"],
            ["She saw a bright star in the sky.", "She saw a bright star at the awards show.", "She observed a luminous celestial object."],
            ["People are shopping.", "Numerous customers browsing for produce in a market", "People are showering."],
            ["There's a red bus making a left turn into a traffic circle that has a sprinkler system.", "A red bus making a turn", "A red bus backing up into a spot"],
            ["She trained a neural network to recognize faces.", "She developed an AI to identify human features.", "She trained a new recruit to recognize faces."],
        ],
        inputs=[textbox0, textbox1, textbox2],
    )

    btn_list = [leftvote_btn, rightvote_btn, tie_btn, bothbad_btn, clear_btn]

    draw_btn.click(
        models.sts_draw,
        inputs=None,
        outputs=[textbox0, textbox1, textbox2],
        api_name="draw_btn_anon"
    )

    send_btn.click(
        check_input_sts,
        inputs=[textbox0, textbox1, textbox2],
        outputs=None
    ).success(
        partial(disable_btns, 5),
        inputs=None,
        outputs=[send_btn, draw_btn, textbox0, textbox1, textbox2],   
    ).success(
        gen_func,
        inputs=[state0, state1, textbox0, textbox1, textbox2, model_selector_left, model_selector_right],
        outputs=[state0, state1, chatbot_left, chatbot_right, model_selector_left, model_selector_right],
        api_name="send_btn_anon"
    ).success(
        enable_btns,
        inputs=None,
        outputs=btn_list
    )

    clear_btn.click(
        clear_history_side_by_side_anon_sts,
        inputs=None,
        outputs=[state0, state1, textbox0, textbox1, textbox2, chatbot_left, chatbot_right, model_selector_left, model_selector_right],
        api_name="clear_btn_anon"
    ).then(
        disable_buttons_side_by_side,
        inputs=None,
        outputs=btn_list
    ).then(
        partial(enable_btns, 5),
        inputs=None,
        outputs=[send_btn, draw_btn, textbox0, textbox1, textbox2],
    )

    dummy_left_model = gr.State("")
    dummy_right_model = gr.State("")
    leftvote_btn.click(
        partial(vote_last_response_sts, "leftvote"),
        inputs=[state0, state1, dummy_left_model, dummy_right_model],
        outputs=[leftvote_btn, rightvote_btn, tie_btn, bothbad_btn, model_selector_left, model_selector_right]
    )
    rightvote_btn.click(
        partial(vote_last_response_sts, "rightvote"),
        inputs=[state0, state1, dummy_left_model, dummy_right_model],
        outputs=[leftvote_btn, rightvote_btn, tie_btn, bothbad_btn, model_selector_left, model_selector_right]
    )
    tie_btn.click(
        partial(vote_last_response_sts, "tievote"),
        inputs=[state0, state1, dummy_left_model, dummy_right_model],
        outputs=[leftvote_btn, rightvote_btn, tie_btn, bothbad_btn, model_selector_left, model_selector_right]
    )
    bothbad_btn.click(
        partial(vote_last_response_sts, "bothbadvote"),
        inputs=[state0, state1, dummy_left_model, dummy_right_model],
        outputs=[leftvote_btn, rightvote_btn, tie_btn, bothbad_btn, model_selector_left, model_selector_right]
    )

def build_side_by_side_ui_named_sts(models):
    notice_markdown = """
# ⚔️ MTEB Arena: STS ☘️
## 📜 Rules (Play the video for an explanation ➡️)
- Input three different texts to two anonymous models and vote which visualizes their similarity better.
- The closer (smaller distance) two corners of the triangle are, the more similar their texts are to the model.
- The distances between the corners are inverted and scaled cosine similarities and displayed on the edges.

## 👇 Vote now!
"""
    model_list = models.models_sts

    state0 = gr.State()
    state1 = gr.State()
    gen_func = partial(sts_side_by_side, models.sts_parallel)
    with gr.Row():
        with gr.Column(scale=3):
            gr.Markdown(notice_markdown, elem_id="notice_markdown")
        with gr.Column(scale=2):
            gr.Video("videos/sts_explanation.mp4", label="Video Explanation", elem_id="video")

    with gr.Group(elem_id="share-region-named"):
        with gr.Row():
            with gr.Column():
                model_selector_left = gr.Dropdown(
                    choices=model_list,
                    value=DEFAULT_MODEL_A,
                    interactive=True,
                    show_label=False,
                    container=False,
                )
            with gr.Column():
                model_selector_right = gr.Dropdown(
                    choices=model_list,
                    value=DEFAULT_MODEL_B,
                    interactive=True,
                    show_label=False,
                    container=False,
                )
        with gr.Row():
            with gr.Accordion("🔍 Expand to see all model descriptions", open=False):
                model_description_md = models.get_model_description_md(task_type="sts")
                gr.Markdown(model_description_md, elem_id="model_description_markdown")

        with gr.Row():
            with gr.Column():
                chatbot_left = gr.Plot(label="Model A")
            with gr.Column():
                chatbot_right = gr.Plot(label="Model B")
        with gr.Row():
            leftvote_btn = gr.Button(
                value="👈  A is better", visible=False, interactive=False
            )
            rightvote_btn = gr.Button(
                value="👉  B is better", visible=False, interactive=False
            )
            tie_btn = gr.Button(value="🤝  Tie", visible=False, interactive=False)
            bothbad_btn = gr.Button(
                value="👎  Both are bad", visible=False, interactive=False
            )

    with gr.Row():
        textbox0 = gr.Textbox(
            show_label=True,
            label="Text (1)",
            placeholder="👉 Enter text",
            container=True,
            elem_id="input_box",
        )
        textbox1 = gr.Textbox(
            show_label=True,
            label="Text (2)",
            placeholder="👉 Enter text",
            container=True,
            elem_id="input_box",
        )
        textbox2 = gr.Textbox(
            show_label=True,
            label="Text (3)",
            placeholder="👉 Enter text",
            container=True,
            elem_id="input_box",
        )
        send_btn = gr.Button(value="Send", variant="primary", scale=0)
        draw_btn = gr.Button(value="🎲 Random sample", variant="primary", scale=0)

    with gr.Row():
        clear_btn = gr.Button(value="🗑️  Clear history", interactive=False)

    gr.Examples(
        examples=[
            ["hello", "good morning", "早上好"],
            ["She saw a bright star in the sky.", "She saw a bright star at the awards show.", "She observed a luminous celestial object."],
            ["People are shopping.", "Numerous customers browsing for produce in a market", "People are showering."],
            ["There's a red bus making a left turn into a traffic circle that has a sprinkler system.", "A red bus making a turn", "A red bus backing up into a spot"],
            ["She trained a neural network to recognize faces.", "She developed an AI to identify human features.", "She trained a new recruit to recognize faces."],
        ],
        inputs=[textbox0, textbox1, textbox2],
    )
    
    btn_list = [leftvote_btn, rightvote_btn, tie_btn, bothbad_btn, clear_btn]

    draw_btn.click(
        models.sts_draw,
        inputs=None,
        outputs=[textbox0, textbox1, textbox2],
    )

    send_btn.click(
        check_input_sts,
        inputs=[textbox0, textbox1, textbox2],
        outputs=None
    ).success(
        partial(disable_btns, 5),
        inputs=None,
        outputs=[send_btn, draw_btn, textbox0, textbox1, textbox2],
    ).success(
        gen_func,
        inputs=[state0, state1, textbox0, textbox1, textbox2, model_selector_left, model_selector_right],
        outputs=[state0, state1, chatbot_left, chatbot_right],
        api_name="send_side_by_side"
    ).success(
        enable_btns,
        inputs=None,
        outputs=btn_list
    )

    clear_btn.click(
        clear_history_side_by_side_anon_sts,
        inputs=None,
        outputs=[state0, state1, textbox0, textbox1, textbox2, chatbot_left, chatbot_right],
        api_name="clear_btn_anon"
    ).then(
        disable_buttons_side_by_side,
        inputs=None,
        outputs=btn_list
    ).then(
        partial(enable_btns, 5),
        inputs=None,
        outputs=[send_btn, draw_btn, textbox0, textbox1, textbox2],
    )

    leftvote_btn.click(
        partial(vote_last_response_sts, "leftvote"),
        inputs=[state0, state1, model_selector_left, model_selector_right],
        outputs=[leftvote_btn, rightvote_btn, tie_btn, bothbad_btn, model_selector_left, model_selector_right]
    )
    rightvote_btn.click(
        partial(vote_last_response_sts, "rightvote"),
        inputs=[state0, state1, model_selector_left, model_selector_right],
        outputs=[leftvote_btn, rightvote_btn, tie_btn, bothbad_btn, model_selector_left, model_selector_right]
    )
    tie_btn.click(
        partial(vote_last_response_sts, "tievote"),
        inputs=[state0, state1, model_selector_left, model_selector_right],
        outputs=[leftvote_btn, rightvote_btn, tie_btn, bothbad_btn, model_selector_left, model_selector_right]
    )
    bothbad_btn.click(
        partial(vote_last_response_sts, "bothbadvote"),
        inputs=[state0, state1, model_selector_left, model_selector_right],
        outputs=[leftvote_btn, rightvote_btn, tie_btn, bothbad_btn, model_selector_left, model_selector_right]
    )

def build_single_model_ui_sts(models):
    notice_markdown = f"""
# 💧 MTEB Arena Single Model: STS ☘️
"""
    #| [GitHub](https://github.com/embeddings-benchmark) |
    ## 🤖 Choose any model

    state = gr.State()
    gen_func = partial(sts, models.sts)
    gr.Markdown(notice_markdown, elem_id="notice_markdown")

    model_list = models.models_sts

    with gr.Row(elem_id="model_selector_row"):
        model_selector = gr.Dropdown(
            choices=model_list,
            value=DEFAULT_MODEL_A,
            interactive=True,
            show_label=False
        )

    with gr.Row():
        with gr.Accordion(
            "🔍 Expand to see all model descriptions",
            open=False,
            elem_id="model_description_accordion",
        ):
            model_description_md = models.get_model_description_md(task_type="sts")
            gr.Markdown(model_description_md, elem_id="model_description_markdown")

    with gr.Group(elem_id="model"):
        with gr.Row():
            chatbot = gr.Plot(label="Model")

    with gr.Row():
        textbox0 = gr.Textbox(
            show_label=True,
            label="Text (1)",
            placeholder="👉 Enter text",
            container=True,
            elem_id="input_box",
        )
        textbox1 = gr.Textbox(
            show_label=True,
            label="Text (2)",
            placeholder="👉 Enter text",
            container=True,
            elem_id="input_box",
        )
        textbox2 = gr.Textbox(
            show_label=True,
            label="Text (3)",
            placeholder="👉 Enter text",
            container=True,
            elem_id="input_box",
        )
        send_btn = gr.Button(value="Send", variant="primary", scale=0)
        draw_btn = gr.Button(value="🎲 Random sample", variant="primary", scale=0)

    with gr.Row() as button_row:
        upvote_btn = gr.Button(value="👍  Upvote", interactive=False)
        downvote_btn = gr.Button(value="👎  Downvote", interactive=False)
        flag_btn = gr.Button(value="⚠️  Flag", interactive=False)
        clear_btn = gr.Button(value="🗑️  Clear history", interactive=False)

    gr.Examples(
        examples=[
            ["hello", "good morning", "早上好"],
            ["She saw a bright star in the sky.", "She saw a bright star at the awards show.", "She observed a luminous celestial object."],
            ["People are shopping.", "Numerous customers browsing for produce in a market", "People are showering."],
            ["There's a red bus making a left turn into a traffic circle that has a sprinkler system.", "A red bus making a turn", "A red bus backing up into a spot"],
            ["She trained a neural network to recognize faces.", "She developed an AI to identify human features.", "She trained a new recruit to recognize faces."],
        ],
        inputs=[textbox0, textbox1, textbox2],
    )
    
    btn_list = [upvote_btn, downvote_btn, flag_btn, clear_btn]

    draw_btn.click(
        models.sts_draw,
        inputs=None,
        outputs=[textbox0, textbox1, textbox2],
    )

    model_selector.change(
        clear_history_sts, 
        inputs=None, 
        outputs=[state, textbox0, textbox1, textbox2, chatbot], 
        api_name="model_selector_single"
    ).then(
        partial(disable_btns, 4),
        inputs=None,
        outputs=btn_list
    ).then(
        partial(enable_btns, 5),
        inputs=None,
        outputs=[send_btn, draw_btn, textbox0, textbox1, textbox2],
    )

    send_btn.click(
        check_input_sts,
        inputs=[textbox0, textbox1, textbox2],
        outputs=None
    ).success(
        partial(disable_btns, 5),
        inputs=None,
        outputs=[send_btn, draw_btn, textbox0, textbox1, textbox2],   
    ).success(
        gen_func,
        inputs=[state, textbox0, textbox1, textbox2, model_selector],
        outputs=[state, chatbot],
        api_name="send_btn_single",
        show_progress="full"
    ).success(
        partial(enable_btns, 4),
        inputs=None,
        outputs=btn_list
    )

    upvote_btn.click(
        partial(vote_last_response_single_sts, "upvote"),
        inputs=[state, model_selector],
        outputs=[upvote_btn, downvote_btn, flag_btn]
    )
    downvote_btn.click(
        partial(vote_last_response_single_sts, "downvote"),
        inputs=[state, model_selector],
        outputs=[upvote_btn, downvote_btn, flag_btn]
    )
    flag_btn.click(
        partial(vote_last_response_single_sts, "flag"),
        inputs=[state, model_selector],
        outputs=[upvote_btn, downvote_btn, flag_btn]
    )
    clear_btn.click(
        clear_history_sts,
        inputs=None,
        outputs=[state, textbox0, textbox1, textbox2, chatbot],
        api_name="clear_history_single",
        show_progress="full"
    ).then(
        partial(disable_btns, 4),
        inputs=None,
        outputs=btn_list
    ).then(
        partial(enable_btns, 5),
        inputs=None,
        outputs=[send_btn, draw_btn, textbox0, textbox1, textbox2],
    )
