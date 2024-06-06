from functools import partial
import datetime
import json
import time
import os
import uuid

import gradio as gr

from log_utils import build_logger

LOGDIR = os.getenv("LOGDIR", "./MTEB-Arena-logs/vote_log")

acknowledgment_md = """
### Acknowledgment
We thank X, Y, Z for their generous sponsorship. If you would like to sponsor us, please get in touch.

We thank [Chatbot Arena](https://chat.lmsys.org/), [Vision Arena](https://huggingface.co/spaces/WildVision/vision-arena) and [GenAI-Arena](https://huggingface.co/spaces/TIGER-Lab/GenAI-Arena) for their great work.
"""
# loggers for side-by-side and battle
retrieval_logger = build_logger("gradio_retrieval", "gradio_retrieval.log")
clustering_logger = build_logger("gradio_clustering", "gradio_clustering.log")
sts_logger = build_logger("gradio_sts", "gradio_sts.log")

disable_btn = gr.update(interactive=False)
disable_btn_visible = gr.update(interactive=False, visible=False)

def get_ip(request: gr.Request):
    if request:
        if "cf-connecting-ip" in request.headers:
            ip = request.headers["cf-connecting-ip"] or request.client.host
        else:
            ip = request.client.host
    else:
        ip = None
    return ip

def clear_history():
    return None, "", None

def clear_history_sts():
    return None, "", "", "", None

def clear_history_side_by_side():
    return None, None, "", None, None

def clear_history_side_by_side_anon():
    return None, None, "", None, None, gr.Markdown("", visible=False), gr.Markdown("", visible=False)

def clear_history_side_by_side_anon_sts():
    return None, None, "", "", "",  None, None, gr.Markdown("", visible=False), gr.Markdown("", visible=False)

def enable_buttons(i=5):
    return tuple(gr.update(interactive=True) for _ in range(i))

def disable_buttons(i=5):
    return tuple(gr.update(interactive=False) for _ in range(i))

def disable_buttons_side_by_side(i=6):
    return tuple(gr.update(visible=i>=4, interactive=False) for i in range(i))

def enable_buttons_side_by_side(i=6):
    return tuple(gr.update(visible=True, interactive=True) for i in range(i))

def enable_buttons_side_by_side_clustering(state0):
    if (state0 is not None) and (len(state0.prompts) >= 3):
        return enable_buttons_side_by_side(7)
    else:
        return enable_buttons_side_by_side(2) + disable_buttons_side_by_side(5)

def vote_last_response(vote_type, state0, state1, model_selector0, model_selector1, request: gr.Request):
    retrieval_logger.info(f"{vote_type} (named). ip: {get_ip(request)}")
    with open(get_conv_log_filename(), "a") as fout:
        data = {
            "tstamp": round(time.time(), 4),
            "task_type": "retrieval",
            "type": vote_type,
            "models": [model_selector0, model_selector1],
            "states": [state0.dict(), state1.dict()],
            "ip": get_ip(request),
        }
        fout.write(json.dumps(data) + "\n")

    if vote_type == "share": return

    if model_selector0 == "":
        return ("",) + (disable_btn,) * 4 + (gr.Markdown(f"### Model A: {state0.model_name}", visible=True), gr.Markdown(f"### Model B: {state1.model_name}", visible=True))
    return ("",) + (disable_btn,) * 4 + (gr.Markdown(state0.model_name, visible=True), gr.Markdown(state1.model_name, visible=True))

def vote_last_response_sts(vote_type, state0, state1, model_selector0, model_selector1, request: gr.Request):
    sts_logger.info(f"{vote_type} (named). ip: {get_ip(request)}")
    with open(get_conv_log_filename(), "a") as fout:
        data = {
            "tstamp": round(time.time(), 4),
            "task_type": "sts",
            "type": vote_type,
            "models": [model_selector0, model_selector1],
            "states": [state0.dict(), state1.dict()],
            "ip": get_ip(request),
        }
        fout.write(json.dumps(data) + "\n")

    if vote_type == "share": return

    if model_selector0 == "":
        return (disable_btn,) * 4 + (gr.Markdown(f"### Model A: {state0.model_name}", visible=True), gr.Markdown(f"### Model B: {state1.model_name}", visible=True))
    return (disable_btn,) * 4 + (gr.Markdown(state0.model_name, visible=True), gr.Markdown(state1.model_name, visible=True))

def vote_last_response_clustering(vote_type, state0, state1, model_selector0, model_selector1, request: gr.Request):
    clustering_logger.info(f"{vote_type} (named). ip: {get_ip(request)}")
    with open(get_conv_log_filename(), "a") as fout:
        data = {
            "tstamp": round(time.time(), 4),
            "task_type": "clustering",
            "type": vote_type,
            "models": [model_selector0, model_selector1],
            "states": [state0.dict(), state1.dict()],
            "ip": get_ip(request),
        }
        fout.write(json.dumps(data) + "\n")

    if vote_type == "share": return

    if model_selector0 == "":
        return (disable_btn_visible,) * 2 + (disable_btn,) * 4 + (gr.Markdown(f"### Model A: {state0.model_name}", visible=True), gr.Markdown(f"### Model B: {state1.model_name}", visible=True))
    return (disable_btn_visible,) * 2 + (disable_btn,) * 4 + (gr.Markdown(state0.model_name, visible=True), gr.Markdown(state1.model_name, visible=True))

def vote_last_response_single(vote_type, state, model_selector, request: gr.Request):
    retrieval_logger.info(f"{vote_type} (named). ip: {get_ip(request)}")
    with open(get_conv_log_filename(), "a") as fout:
        data = {
            "tstamp": round(time.time(), 4),
            "task_type": "retrieval",
            "type": vote_type,
            "models": model_selector,
            "states": state.dict(),
            "ip": get_ip(request),
        }
        fout.write(json.dumps(data) + "\n")
    return ("",) + (disable_btn,) * 3

def vote_last_response_single_sts(vote_type, state, model_selector, request: gr.Request):
    sts_logger.info(f"{vote_type} (named). ip: {get_ip(request)}")
    with open(get_conv_log_filename(), "a") as fout:
        data = {
            "tstamp": round(time.time(), 4),
            "task_type": "sts",
            "type": vote_type,
            "models": model_selector,
            "states": state.dict(),
            "ip": get_ip(request),
        }
        fout.write(json.dumps(data) + "\n")
    return (disable_btn,) * 3

def vote_last_response_single_clustering(vote_type, state, model_selector, request: gr.Request):
    clustering_logger.info(f"{vote_type} (named). ip: {get_ip(request)}")
    with open(get_conv_log_filename(), "a") as fout:
        data = {
            "tstamp": round(time.time(), 4),
            "task_type": "clustering",
            "type": vote_type,
            "models": model_selector,
            "states": state.dict(),
            "ip": get_ip(request),
        }
        fout.write(json.dumps(data) + "\n")
    return (disable_btn_visible,) * 2 + (disable_btn,) * 3

def get_conv_log_filename():
    t = datetime.datetime.now()
    name = os.path.join(LOGDIR, f"{t.year}-{t.month:02d}-{t.day:02d}-conv.json")
    return name

class RetrievalState:
    def __init__(self, model_name):
        self.conv_id = uuid.uuid4().hex
        self.model_name = model_name
        self.prompt = None
        self.output = None

    def dict(self):
        return {"conv_id": self.conv_id, "model_name": self.model_name, "prompt": self.prompt, "output": self.output}

def retrieve_side_by_side(gen_func, state0, state1, text, model_name0, model_name1, request: gr.Request):
    if not text: raise gr.Warning("Prompt cannot be empty.")
    state0, state1 = RetrievalState(model_name0), RetrievalState(model_name1)
    ip = get_ip(request)
    retrieval_logger.info(f"Retrieval. ip: {ip}")
    start_tstamp = time.time()
    model_name0, model_name1 = "", ""
    retrieved_txt0, retrieved_txt1, model_name0, model_name1 = gen_func(text, model_name0, model_name1)
    state0.prompt, state1.prompt = text, text
    state0.output, state1.output = retrieved_txt0, retrieved_txt1
    state0.model_name, state1.model_name = model_name0, model_name1
    
    yield state0, state1, retrieved_txt0, retrieved_txt1, \
        gr.Markdown(f"### Model A: {model_name0}", visible=False), gr.Markdown(f"### Model B: {model_name1}", visible=False)
    
    finish_tstamp = time.time()
    
    with open(get_conv_log_filename(), "a") as fout:
        data = {
            "tstamp": round(finish_tstamp, 4),
            "task_type": "retrieval",
            "type": "chat",
            "model": model_name0,
            "gen_params": {},
            "start": round(start_tstamp, 4),
            "finish": round(finish_tstamp, 4),
            "state": state0.dict(),
            "ip": get_ip(request),
        }
        fout.write(json.dumps(data) + "\n")
        # append_json_item_on_log_server(data, get_conv_log_filename())
        data = {
            "tstamp": round(finish_tstamp, 4),
            "task_type": "retrieval",
            "type": "chat",
            "model": model_name1,
            "gen_params": {},
            "start": round(start_tstamp, 4),
            "finish": round(finish_tstamp, 4),
            "state": state1.dict(),
            "ip": get_ip(request),
        }
        fout.write(json.dumps(data) + "\n")
        # append_json_item_on_log_server(data, get_conv_log_filename())

def retrieve(gen_func, state, text, model_name, request: gr.Request):
    if not text: raise gr.Warning("Prompt cannot be empty.")
    if not model_name: raise gr.Warning("Model name cannot be empty.")
    state = RetrievalState(model_name)
    ip = get_ip(request)
    retrieval_logger.info(f"Retrieval. ip: {ip}")
    start_tstamp = time.time()
    retrieved_txt = gen_func(text, model_name)
    state.prompt = text
    state.output = retrieved_txt
    state.model_name = model_name

    yield state, retrieved_txt
    
    finish_tstamp = time.time()
    # logger.info(f"===output===: {output}")

    with open(get_conv_log_filename(), "a") as fout:
        data = {
            "tstamp": round(finish_tstamp, 4),
            "task_type": "retrieval",
            "type": "chat",
            "model": model_name,
            "gen_params": {},
            "start": round(start_tstamp, 4),
            "finish": round(finish_tstamp, 4),
            "state": state.dict(),
            "ip": get_ip(request),
        }
        fout.write(json.dumps(data) + "\n")
    #    append_json_item_on_log_server(data, get_conv_log_filename())

def check_input_retrieval(txt):
    if not(txt): raise gr.Warning("Prompt cannot be empty.")

def build_side_by_side_ui_anon(models):
    notice_markdown = """
# ⚔️ MTEB Arena ⚔️ : Retrieval 🔎
## 📜 Rules
- Input a search query to two anonymous models and vote which retrieves the better passage for your query.
- Whenever you have decided which model is better, click the button below to vote.
- Click "New Round" to start a new round.

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

    with gr.Row():
        textbox = gr.Textbox(
            show_label=False,
            placeholder="👉 Enter your query and press ENTER",
            container=True,
            elem_id="input_box",
        )
        send_btn = gr.Button(value="Send", variant="primary", scale=0)

    with gr.Row():
        clear_btn = gr.Button(value="🎲 New Round", interactive=False)
        # regenerate_btn = gr.Button(value="🔄  Regenerate", interactive=False)
        share_btn = gr.Button(value="📷  Share")

    gr.Markdown(acknowledgment_md, elem_id="ack_markdown")

    btn_list = [leftvote_btn, rightvote_btn, tie_btn, bothbad_btn, clear_btn,]

    textbox.submit(
        check_input_retrieval,
        inputs=textbox,
        outputs=None,
    ).success(
        partial(disable_buttons_side_by_side, 2),
        inputs=None,
        outputs=[send_btn, textbox],
    ).then(
        gen_func,
        inputs=[state0, state1, textbox, model_selector_left, model_selector_right],
        outputs=[state0, state1, chatbot_left, chatbot_right, model_selector_left, model_selector_right],
        api_name="submit_btn_anon"
    ).then(
        enable_buttons_side_by_side,
        inputs=None,
        outputs=btn_list
    )

    send_btn.click(
        check_input_retrieval,
        inputs=textbox,
        outputs=None,
    ).success(        
        partial(disable_buttons_side_by_side, 2),
        inputs=None,
        outputs=[send_btn, textbox],   
    ).then(
        gen_func,
        inputs=[state0, state1, textbox, model_selector_left, model_selector_right],
        outputs=[state0, state1, chatbot_left, chatbot_right, model_selector_left, model_selector_right],
        api_name="send_btn_anon"
    ).then(
        enable_buttons_side_by_side,
        inputs=state0,
        outputs=btn_list
    )

    clear_btn.click(
        clear_history_side_by_side_anon,
        inputs=None,
        outputs=[state0, state1, textbox, chatbot_left, chatbot_right, model_selector_left, model_selector_right],
        api_name="clear_btn_anon"
    ).then(
        disable_buttons_side_by_side,
        inputs=None,
        outputs=btn_list
    ).then(
        partial(enable_buttons_side_by_side, 2),
        inputs=None,
        outputs=[send_btn, textbox],
    )
    
    """
    regenerate_btn.click(
        gen_func,
        inputs=[state0, state1, textbox, model_selector_left, model_selector_right],
        outputs=[state0, state1, chatbot_left, chatbot_right, model_selector_left, model_selector_right],
        api_name="regenerate_btn_anon"
    ).then(
        enable_buttons_side_by_side,
        inputs=None,
        outputs=btn_list
    )
    """
    dummy_left_model = gr.State("")
    dummy_right_model = gr.State("")
    leftvote_btn.click(
        partial(vote_last_response, "leftvote"),
        inputs=[state0, state1, dummy_left_model, dummy_right_model],
        outputs=[textbox, leftvote_btn, rightvote_btn, tie_btn, bothbad_btn, model_selector_left, model_selector_right]
    )
    rightvote_btn.click(
        partial(vote_last_response, "rightvote"),
        inputs=[state0, state1, dummy_left_model, dummy_right_model],
        outputs=[textbox, leftvote_btn, rightvote_btn, tie_btn, bothbad_btn, model_selector_left, model_selector_right]
    )
    tie_btn.click(
        partial(vote_last_response, "tievote"),
        inputs=[state0, state1, dummy_left_model, dummy_right_model],
        outputs=[textbox, leftvote_btn, rightvote_btn, tie_btn, bothbad_btn, model_selector_left, model_selector_right]
    )
    bothbad_btn.click(
        partial(vote_last_response, "bothbadvote"),
        inputs=[state0, state1, dummy_left_model, dummy_right_model],
        outputs=[textbox, leftvote_btn, rightvote_btn, tie_btn, bothbad_btn, model_selector_left, model_selector_right]
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
# ⚔️ MTEB Arena ⚔️ : Retrieval 🔎

## 📜 Rules
- Input a search query to two anonymous models and vote which retrieves the better passage for your query.
- Whenever you have decided which model is better, click the button below to vote.
- Click "New Round" to start a new round.

## 👇 Choose two models & vote now!
"""
    model_list = list(models.model_meta.keys())

    state0 = gr.State()
    state1 = gr.State()
    gen_func = partial(retrieve_side_by_side, models.retrieve_parallel)
    gr.Markdown(notice_markdown, elem_id="notice_markdown")

    with gr.Group(elem_id="share-region-named"):
        with gr.Row():
            with gr.Column():
                model_selector_left = gr.Dropdown(
                    choices=model_list,
                    value=model_list[0] if len(model_list) > 0 else "",
                    interactive=True,
                    show_label=False,
                    container=False,
                    allow_custom_value=True
                    )
            with gr.Column():
                model_selector_right = gr.Dropdown(
                    choices=model_list,
                    value=model_list[1] if len(model_list) > 1 else "",
                    interactive=True,
                    show_label=False,
                    container=False,
                    allow_custom_value=True
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

    with gr.Row():
        textbox = gr.Textbox(
            show_label=False,
            placeholder="👉 Enter your query and press ENTER",
            elem_id="input_box"
        )
        send_btn = gr.Button(value="Send", variant="primary", scale=0)

    with gr.Row():
        clear_btn = gr.Button(value="🗑️  Clear history", interactive=False)
        share_btn = gr.Button(value="📷  Share")

    gr.Markdown(acknowledgment_md, elem_id="ack_markdown")
    
    btn_list = [leftvote_btn, rightvote_btn, tie_btn, bothbad_btn, clear_btn]

    textbox.submit(
        check_input_retrieval,
        inputs=textbox,
        outputs=None,
    ).success(        
        partial(disable_buttons_side_by_side, 2),
        inputs=None,
        outputs=[send_btn, textbox],
    ).then(        
        gen_func,
        inputs=[state0, state1, textbox, model_selector_left, model_selector_right],
        outputs=[state0, state1, chatbot_left, chatbot_right],
        api_name="textbox_side_by_side"
    ).then(
        enable_buttons_side_by_side, 
        inputs=None,  
        outputs=btn_list 
    )
    
    send_btn.click(
        check_input_retrieval,
        inputs=textbox,
        outputs=None,
    ).success(
        partial(disable_buttons_side_by_side, 2),
        inputs=None,
        outputs=[send_btn, textbox],
    ).then(
        gen_func,
        inputs=[state0, state1, textbox, model_selector_left, model_selector_right],
        outputs=[state0, state1, chatbot_left, chatbot_right],
        api_name="send_side_by_side"
    ).then(
        enable_buttons_side_by_side,
        inputs=None,
        outputs=btn_list
    )
    
    clear_btn.click(
        clear_history_side_by_side, 
        inputs=None, 
        outputs=[state0, state1, textbox, chatbot_left, chatbot_right], 
        api_name="clear_btn_side_by_side"
    ).then(
        disable_buttons_side_by_side,
        inputs=None,
        outputs=btn_list
    ).then(
        partial(enable_buttons_side_by_side, 2),
        inputs=None,
        outputs=[send_btn, textbox],
    )

    leftvote_btn.click(
        partial(vote_last_response, "leftvote"),
        inputs=[state0, state1, model_selector_left, model_selector_right],
        outputs=[textbox, leftvote_btn, rightvote_btn, tie_btn, bothbad_btn, model_selector_left, model_selector_right]
    )
    rightvote_btn.click(
        partial(vote_last_response, "rightvote"),
        inputs=[state0, state1, model_selector_left, model_selector_right],
        outputs=[textbox, leftvote_btn, rightvote_btn, tie_btn, bothbad_btn, model_selector_left, model_selector_right]
    )
    tie_btn.click(
        partial(vote_last_response, "tievote"),
        inputs=[state0, state1, model_selector_left, model_selector_right],
        outputs=[textbox, leftvote_btn, rightvote_btn, tie_btn, bothbad_btn, model_selector_left, model_selector_right]
    )
    bothbad_btn.click(
        partial(vote_last_response, "bothbadvote"),
        inputs=[state0, state1, model_selector_left, model_selector_right],
        outputs=[textbox, leftvote_btn, rightvote_btn, tie_btn, bothbad_btn, model_selector_left, model_selector_right]
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
# MTEB Arena Playground: Retrieval 🔎
"""
    #| [GitHub](https://github.com/embeddings-benchmark) |
    ### 🤖 Choose any retriever
    state = gr.State()
    gen_func = partial(retrieve, models.retrieve)
    gr.Markdown(notice_markdown, elem_id="notice_markdown")

    model_list = list(models.model_meta.keys())

    with gr.Row(elem_id="model_selector_row"):
        model_selector = gr.Dropdown(
            choices=model_list,
            value=model_list[0] if len(model_list) > 0 else "",
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
        textbox = gr.Textbox(
            show_label=False,
            placeholder="👉 Enter your query and press ENTER",
            elem_id="input_box"
        )

        send_btn = gr.Button(value="Send", variant="primary", scale=0)

    with gr.Row():
        chatbot = gr.Chatbot(
            label="Model",
            elem_id="chatbot",
            height=550,
            show_copy_button=True,
        )

    with gr.Row() as button_row:
        upvote_btn = gr.Button(value="👍  Upvote", interactive=False)
        downvote_btn = gr.Button(value="👎  Downvote", interactive=False)
        flag_btn = gr.Button(value="⚠️  Flag", interactive=False)
        clear_btn = gr.Button(value="🗑️  Clear history", interactive=False)

    gr.Markdown(acknowledgment_md, elem_id="ack_markdown")
    
    btn_list = [upvote_btn, downvote_btn, flag_btn, clear_btn]

    model_selector.change(
        clear_history, 
        inputs=None,
        outputs=[state, textbox, chatbot], 
        api_name="model_selector_single"
    ).then(
        disable_buttons,
        inputs=None,
        outputs=btn_list
    ).then(
        partial(enable_buttons_side_by_side, 2),
        inputs=None,
        outputs=[send_btn, textbox],
    )

    textbox.submit(
        check_input_retrieval,
        inputs=textbox,
        outputs=None,
    ).success(
        partial(disable_buttons_side_by_side, 2),
        inputs=None,
        outputs=[send_btn, textbox],
    ).then(        
        gen_func,
        inputs=[state, textbox, model_selector], 
        outputs=[state, chatbot],
        api_name="submit_btn_single",
        show_progress = "full"
    ).success(
        enable_buttons, 
        inputs=None,  
        outputs=btn_list 
    )

    send_btn.click(
        check_input_retrieval,
        inputs=textbox,
        outputs=None,
    ).success(
        partial(disable_buttons_side_by_side, 2),
        inputs=None,
        outputs=[send_btn, textbox],
    ).then(
        gen_func,
        inputs=[state, textbox, model_selector],
        outputs=[state, chatbot],
        api_name="send_btn_single",
        show_progress = "full"
    ).success(
        enable_buttons,
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
        disable_buttons,
        inputs=None,
        outputs=btn_list
    ).then(
        partial(enable_buttons_side_by_side, 2),
        inputs=None,
        outputs=[send_btn, textbox],
    )

### Clustering ###

def check_input_clustering(txt):
    if not(txt): raise gr.Warning("Prompt cannot be empty.")

# https://github.com/lm-sys/FastChat/blob/73936244535664c7e4c9bc1a419aa7f77b2da88e/fastchat/serve/gradio_web_server.py#L100
# https://github.com/lm-sys/FastChat/blob/73936244535664c7e4c9bc1a419aa7f77b2da88e/fastchat/serve/gradio_block_arena_named.py#L165
class ClusteringState:
    def __init__(self, model_name):
        self.conv_id = uuid.uuid4().hex
        self.model_name = model_name
        self.prompts = []
        self.output = None

    def dict(self):
        return {"conv_id": self.conv_id, "model_name": self.model_name, "prompt": self.prompts, "output": self.output}

def clustering_side_by_side(gen_func, state0, state1, txt, model_name0, model_name1, request: gr.Request):
    if not txt: raise gr.Warning("Prompt cannot be empty.")
    if state0 is None:
        state0 = ClusteringState(model_name1)
    if state1 is None:
        state1 = ClusteringState(model_name0)
    state0.prompts.append(txt)
    state1.prompts.append(txt)

    ip = get_ip(request)
    clustering_logger.info(f"Clustering. ip: {ip}")
    start_tstamp = time.time()
    model_name0, model_name1 = "", ""
    generated_image0, generated_image1, model_name0, model_name1 = gen_func(state0.prompts, model_name0, model_name1)
    #state0.output, state1.output = generated_image0, generated_image1
    state0.model_name, state1.model_name = model_name0, model_name1
    
    yield state0, state1, generated_image0, generated_image1, \
        gr.Markdown(f"### Model A: {model_name0}", visible=False), gr.Markdown(f"### Model B: {model_name1}", visible=False), None
    
    finish_tstamp = time.time()
    
    with open(get_conv_log_filename(), "a") as fout:
        data = {
            "tstamp": round(finish_tstamp, 4),
            "task_type": "clustering",
            "type": "chat",
            "model": model_name0,
            "gen_params": {},
            "start": round(start_tstamp, 4),
            "finish": round(finish_tstamp, 4),
            "state": state0.dict(),
            "ip": get_ip(request),
        }
        fout.write(json.dumps(data) + "\n")
        # append_json_item_on_log_server(data, get_conv_log_filename())
        data = {
            "tstamp": round(finish_tstamp, 4),
            "task_type": "clustering",
            "type": "chat",
            "model": model_name1,
            "gen_params": {},
            "start": round(start_tstamp, 4),
            "finish": round(finish_tstamp, 4),
            "state": state1.dict(),
            "ip": get_ip(request),
        }
        fout.write(json.dumps(data) + "\n")
        # append_json_item_on_log_server(data, get_conv_log_filename())


def clustering(gen_func, state, text, model_name, request: gr.Request):
    if not text: raise gr.Warning("Prompt cannot be empty.")
    if not model_name: raise gr.Warning("Model name cannot be empty.")
    if state is None:
        state = ClusteringState(model_name)
    ip = get_ip(request)
    clustering_logger.info(f"Clustering. ip: {ip}")
    start_tstamp = time.time()
    state.prompts.append(text)
    generated_img = gen_func(state.prompts, model_name)
    #state.output = retrieved_txt
    state.model_name = model_name

    yield state, generated_img, None
    
    finish_tstamp = time.time()
    # logger.info(f"===output===: {output}")

    with open(get_conv_log_filename(), "a") as fout:
        data = {
            "tstamp": round(finish_tstamp, 4),
            "task_type": "clustering",
            "type": "chat",
            "model": model_name,
            "gen_params": {},
            "start": round(start_tstamp, 4),
            "finish": round(finish_tstamp, 4),
            "state": state.dict(),
            "ip": get_ip(request),
        }
        fout.write(json.dumps(data) + "\n")
    #    append_json_item_on_log_server(data, get_conv_log_filename())

def build_side_by_side_ui_anon_clustering(models):
    notice_markdown = """
# ⚔️ MTEB Arena ⚔️ : Clustering ✨

## 📜 Rules
- Input texts one-by-one and submit them to two anonymous models. Vote which model clusters the texts better in the plot.
- You have to **enter at least 3 texts**, else cluster qualities cannot be judged.
- The cluster will be 1D for 1 text, 2D for 2-3 texts, 3D for >3 texts.
- Whenever you have decided which model is better, click the button below to vote.
- Click "New Round" to start a new round.
- Tips for voting: Are similar texts clustered close together? Are dissimilar texts clustered far apart?

## 👇 Choose two models & vote now!
"""
    state0 = gr.State(None)
    state1 = gr.State(None)
    gen_func = partial(clustering_side_by_side, models.clustering_parallel)    

    gr.Markdown(notice_markdown, elem_id="notice_markdown")

    with gr.Group(elem_id="share-region-anon"):
        with gr.Accordion("🔍 Expand to see all Arena players", open=False):
            model_description_md = models.get_model_description_md()
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
            show_label=False,
            placeholder="👉 Enter your text and press ENTER",
            container=True,
            elem_id="input_box",
        )
        send_btn = gr.Button(value="Send", variant="primary", scale=0)

    with gr.Row():
        clear_btn = gr.Button(value="🎲 New Round", interactive=False)

    gr.Markdown(acknowledgment_md, elem_id="ack_markdown")

    btn_list = [leftvote_btn, rightvote_btn, tie_btn, bothbad_btn, clear_btn]

    textbox.submit(
        check_input_clustering,
        inputs=textbox,
        outputs=None,
    ).success(
        partial(disable_buttons_side_by_side, 2),
        inputs=None,
        outputs=[send_btn, textbox],
    ).then(
        gen_func,
        inputs=[state0, state1, textbox, model_selector_left, model_selector_right],
        outputs=[state0, state1, chatbot_left, chatbot_right, model_selector_left, model_selector_right, textbox],
        api_name="submit_btn_anon"
    ).then(
        enable_buttons_side_by_side_clustering,
        inputs=state0,
        outputs=[send_btn, textbox] + btn_list,
    )

    send_btn.click(
        check_input_clustering,
        inputs=textbox,
        outputs=None,
    ).success(        
        partial(disable_buttons_side_by_side, 2),
        inputs=None,
        outputs=[send_btn, textbox],   
    ).then(
        gen_func,
        inputs=[state0, state1, textbox, model_selector_left, model_selector_right],
        outputs=[state0, state1, chatbot_left, chatbot_right, model_selector_left, model_selector_right, textbox],
        api_name="send_btn_anon"
    ).then(
        enable_buttons_side_by_side_clustering,
        inputs=state0,
        outputs=[send_btn, textbox] + btn_list,
    )

    clear_btn.click(
        clear_history_side_by_side_anon,
        inputs=None,
        outputs=[state0, state1, textbox, chatbot_left, chatbot_right, model_selector_left, model_selector_right],
        api_name="clear_btn_anon"
    ).then(
        disable_buttons_side_by_side,
        inputs=None,
        outputs=btn_list
    ).then(
        partial(enable_buttons_side_by_side, 2),
        inputs=None,
        outputs=[send_btn, textbox],
    )

    dummy_left_model = gr.State("")
    dummy_right_model = gr.State("")
    leftvote_btn.click(
        partial(vote_last_response_clustering, "leftvote"),
        inputs=[state0, state1, dummy_left_model, dummy_right_model],
        outputs=[send_btn, textbox, leftvote_btn, rightvote_btn, tie_btn, bothbad_btn, model_selector_left, model_selector_right]
    )
    rightvote_btn.click(
        partial(vote_last_response_clustering, "rightvote"),
        inputs=[state0, state1, dummy_left_model, dummy_right_model],
        outputs=[send_btn, textbox, leftvote_btn, rightvote_btn, tie_btn, bothbad_btn, model_selector_left, model_selector_right]
    )
    tie_btn.click(
        partial(vote_last_response_clustering, "tievote"),
        inputs=[state0, state1, dummy_left_model, dummy_right_model],
        outputs=[send_btn, textbox, leftvote_btn, rightvote_btn, tie_btn, bothbad_btn, model_selector_left, model_selector_right]
    )
    bothbad_btn.click(
        partial(vote_last_response_clustering, "bothbadvote"),
        inputs=[state0, state1, dummy_left_model, dummy_right_model],
        outputs=[send_btn, textbox, leftvote_btn, rightvote_btn, tie_btn, bothbad_btn, model_selector_left, model_selector_right]
    )

def build_side_by_side_ui_named_clustering(models):
    notice_markdown = """
# ⚔️ MTEB Arena ⚔️ : Clustering ✨

## 📜 Rules
- Input texts one-by-one and submit them to two models. Vote which model clusters the texts better in the plot.
- You have to **enter at least 3 texts**, else cluster qualities cannot be judged.
- The cluster will be 1D for 1 text, 2D for 2-3 texts, 3D for >3 texts.
- Whenever you have decided which model is better, click the button below to vote.
- Click "New Round" to start a new round.
- Tips for voting: Are similar texts clustered close together? Are dissimilar texts clustered far apart?

## 👇 Choose two models & vote now!
"""
    model_list = list(models.model_meta.keys())

    state0 = gr.State(None)
    state1 = gr.State(None)
    gen_func = partial(clustering_side_by_side, models.clustering_parallel)    

    gr.Markdown(notice_markdown, elem_id="notice_markdown")

    with gr.Group(elem_id="share-region-named"):
        with gr.Row():
            with gr.Column():
                model_selector_left = gr.Dropdown(
                    choices=model_list,
                    value=model_list[0] if len(model_list) > 0 else "",
                    interactive=True,
                    show_label=False,
                    container=False,
                    allow_custom_value=True
                    )
            with gr.Column():
                model_selector_right = gr.Dropdown(
                    choices=model_list,
                    value=model_list[1] if len(model_list) > 1 else "",
                    interactive=True,
                    show_label=False,
                    container=False,
                    allow_custom_value=True
                    )
        with gr.Row():
            with gr.Accordion("🔍 Expand to see all model descriptions", open=False):
                model_description_md = models.get_model_description_md()
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
            show_label=False,
            placeholder="👉 Enter your text and press ENTER",
            elem_id="input_box"
        )
        send_btn = gr.Button(value="Send", variant="primary", scale=0)

    with gr.Row():
        clear_btn = gr.Button(value="🗑️  Clear history", interactive=False)

    gr.Markdown(acknowledgment_md, elem_id="ack_markdown")
    
    btn_list = [leftvote_btn, rightvote_btn, tie_btn, bothbad_btn, clear_btn]


    textbox.submit(
        check_input_clustering,
        inputs=textbox,
        outputs=None,
    ).success(
        partial(disable_buttons_side_by_side, 2),
        inputs=None,
        outputs=[send_btn, textbox],
    ).then(
        gen_func,
        inputs=[state0, state1, textbox, model_selector_left, model_selector_right],
        outputs=[state0, state1, chatbot_left, chatbot_right, model_selector_left, model_selector_right, textbox],
        api_name="textbox_side_by_side"
    ).then(
        enable_buttons_side_by_side_clustering,
        inputs=state0,
        outputs=[send_btn, textbox] + btn_list,
    )

    send_btn.click(
        check_input_clustering,
        inputs=textbox,
        outputs=None,
    ).success(        
        partial(disable_buttons_side_by_side, 2),
        inputs=None,
        outputs=[send_btn, textbox],   
    ).then(
        gen_func,
        inputs=[state0, state1, textbox, model_selector_left, model_selector_right],
        outputs=[state0, state1, chatbot_left, chatbot_right, model_selector_left, model_selector_right, textbox],
        api_name="send_side_by_side"
    ).then(
        enable_buttons_side_by_side_clustering,
        inputs=state0,
        outputs=[send_btn, textbox] + btn_list,
    )

    clear_btn.click(
        clear_history_side_by_side_anon,
        inputs=None,
        outputs=[state0, state1, textbox, chatbot_left, chatbot_right, model_selector_left, model_selector_right],
        api_name="clear_btn_anon"
    ).then(
        disable_buttons_side_by_side,
        inputs=None,
        outputs=btn_list
    ).then(
        partial(enable_buttons_side_by_side, 2),
        inputs=None,
        outputs=[send_btn, textbox],
    )

    leftvote_btn.click(
        partial(vote_last_response_clustering, "leftvote"),
        inputs=[state0, state1, model_selector_left, model_selector_right],
        outputs=[send_btn, textbox, leftvote_btn, rightvote_btn, tie_btn, bothbad_btn, model_selector_left, model_selector_right]
    )
    rightvote_btn.click(
        partial(vote_last_response_clustering, "rightvote"),
        inputs=[state0, state1, model_selector_left, model_selector_right],
        outputs=[send_btn, textbox, leftvote_btn, rightvote_btn, tie_btn, bothbad_btn, model_selector_left, model_selector_right]
    )
    tie_btn.click(
        partial(vote_last_response_clustering, "tievote"),
        inputs=[state0, state1, model_selector_left, model_selector_right],
        outputs=[send_btn, textbox, leftvote_btn, rightvote_btn, tie_btn, bothbad_btn, model_selector_left, model_selector_right]
    )
    bothbad_btn.click(
        partial(vote_last_response_clustering, "bothbadvote"),
        inputs=[state0, state1, model_selector_left, model_selector_right],
        outputs=[send_btn, textbox, leftvote_btn, rightvote_btn, tie_btn, bothbad_btn, model_selector_left, model_selector_right]
    )

def build_single_model_ui_clustering(models):
    notice_markdown = f"""
# MTEB Arena Playground: Clustering ✨
"""
    # | [GitHub](https://github.com/embeddings-benchmark) | 
    ## 🤖 Choose any clustering model

    state = gr.State(None)
    gen_func = partial(clustering, models.clustering)
    gr.Markdown(notice_markdown, elem_id="notice_markdown")

    model_list = list(models.model_meta.keys())

    with gr.Row(elem_id="model_selector_row"):
        model_selector = gr.Dropdown(
            choices=model_list,
            value=model_list[0] if len(model_list) > 0 else "",
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
        textbox = gr.Textbox(
            show_label=False,
            placeholder="👉 Enter your text and press ENTER",
            elem_id="input_box"
        )

        send_btn = gr.Button(value="Send", variant="primary", scale=0)

    with gr.Row():
        chatbot = gr.Plot(label="Model")

    with gr.Row() as button_row:
        upvote_btn = gr.Button(value="👍  Upvote", interactive=False)
        downvote_btn = gr.Button(value="👎  Downvote", interactive=False)
        flag_btn = gr.Button(value="⚠️  Flag", interactive=False)
        clear_btn = gr.Button(value="🗑️  Clear history", interactive=False)

    gr.Markdown(acknowledgment_md, elem_id="ack_markdown")

    btn_list = [upvote_btn, downvote_btn, flag_btn, clear_btn]

    model_selector.change(
        clear_history, 
        inputs=None, 
        outputs=[state, textbox, chatbot], 
        api_name="model_selector_single"
    ).then(
        disable_buttons,
        inputs=None,
        outputs=btn_list
    ).then(
        partial(enable_buttons_side_by_side, 2),
        inputs=None,
        outputs=[send_btn, textbox],
    )
    
    textbox.submit(
        check_input_clustering,
        inputs=textbox,
        outputs=None,
    ).success(
        partial(disable_buttons_side_by_side, 2),
        inputs=None,
        outputs=[send_btn, textbox],
    ).then(
        gen_func,
        inputs=[state, textbox, model_selector],
        outputs=[state, chatbot, textbox],
        api_name="submit_btn_single",
        show_progress="full"
    ).then(
        enable_buttons_side_by_side_clustering,
        inputs=state,
        outputs=[send_btn, textbox] + btn_list,
    )

    send_btn.click(
        check_input_clustering,
        inputs=textbox,
        outputs=None,
    ).success(
        partial(disable_buttons_side_by_side, 2),
        inputs=None,
        outputs=[send_btn, textbox],
    ).then(
        gen_func,
        inputs=[state, textbox, model_selector],
        outputs=[state, chatbot, textbox],
        api_name="send_btn_single",
        show_progress="full"
    ).then(
        enable_buttons_side_by_side_clustering,
        inputs=state,
        outputs=[send_btn, textbox] + btn_list,
    )

    upvote_btn.click(
        partial(vote_last_response_single_clustering, "upvote"),
        inputs=[state, model_selector],
        outputs=[send_btn, textbox, upvote_btn, downvote_btn, flag_btn]
    )
    downvote_btn.click(
        partial(vote_last_response_single_clustering, "downvote"),
        inputs=[state, model_selector],
        outputs=[send_btn, textbox, upvote_btn, downvote_btn, flag_btn]
    )
    flag_btn.click(
        partial(vote_last_response_single_clustering, "flag"),
        inputs=[state, model_selector],
        outputs=[send_btn, textbox, upvote_btn, downvote_btn, flag_btn]
    )
    clear_btn.click(
        clear_history,
        inputs=None,
        outputs=[state, textbox, chatbot],
        api_name="clear_history_single",
        show_progress="full"
    ).then(
        disable_buttons,
        inputs=None,
        outputs=btn_list
    ).then(
        partial(enable_buttons_side_by_side, 2),
        inputs=None,
        outputs=[send_btn, textbox],
    )

### STS ###
class STSState:
    def __init__(self, model_name):
        self.conv_id = uuid.uuid4().hex
        self.model_name = model_name
        self.txt0 = None
        self.txt1 = None
        self.txt2 = None
        self.output = None

    def dict(self):
        return {"conv_id": self.conv_id, "model_name": self.model_name, "txt0": self.txt0, "txt1": self.txt1, "txt2": self.txt2, "output": self.output}

def sts_side_by_side(gen_func, state0, state1, txt0, txt1, txt2, model_name0, model_name1, request: gr.Request):
    if any([x is None for x in (txt0, txt1, txt2)]): raise gr.Warning("Prompt cannot be empty.")
    state0, state1 = STSState(model_name0), STSState(model_name1)
    ip = get_ip(request)
    retrieval_logger.info(f"Retrieval. ip: {ip}")
    start_tstamp = time.time()
    model_name0, model_name1 = "", ""
    generated_image0, generated_image1, model_name0, model_name1 = gen_func(txt0, txt1, txt2, model_name0, model_name1)
    state0.txt0, state0.txt1, state0.txt2 = txt0, txt1, txt2
    state1.txt0, state1.txt1, state1.txt2 = txt0, txt1, txt2
    state0.output, state1.output = generated_image0, generated_image1
    state0.model_name, state1.model_name = model_name0, model_name1
    
    yield state0, state1, generated_image0, generated_image1, \
        gr.Markdown(f"### Model A: {model_name0}", visible=False), gr.Markdown(f"### Model B: {model_name1}", visible=False)
    
    finish_tstamp = time.time()
    
    with open(get_conv_log_filename(), "a") as fout:
        data = {
            "tstamp": round(finish_tstamp, 4),
            "task_type": "sts",
            "type": "chat",
            "model": model_name0,
            "gen_params": {},
            "start": round(start_tstamp, 4),
            "finish": round(finish_tstamp, 4),
            "state": state0.dict(),
            "ip": get_ip(request),
        }
        fout.write(json.dumps(data) + "\n")
        # append_json_item_on_log_server(data, get_conv_log_filename())
        data = {
            "tstamp": round(finish_tstamp, 4),
            "task_type": "sts",
            "type": "chat",
            "model": model_name1,
            "gen_params": {},
            "start": round(start_tstamp, 4),
            "finish": round(finish_tstamp, 4),
            "state": state1.dict(),
            "ip": get_ip(request),
        }
        fout.write(json.dumps(data) + "\n")
        # append_json_item_on_log_server(data, get_conv_log_filename())

def sts(gen_func, state, txt0, txt1, txt2, model_name, request: gr.Request):
    if any([x is None for x in (txt0, txt1, txt2)]): raise gr.Warning("Prompt cannot be empty.")
    if not model_name: raise gr.Warning("Model name cannot be empty.")
    state = STSState(model_name)
    ip = get_ip(request)
    retrieval_logger.info(f"Retrieval. ip: {ip}")
    start_tstamp = time.time()
    generated_image = gen_func(txt0, txt1, txt2, model_name)
    state.txt0, state.txt1, state.txt2 = txt0, txt1, txt2
    state.output = generated_image
    state.model_name = model_name
    
    yield state, generated_image

    finish_tstamp = time.time()
    
    with open(get_conv_log_filename(), "a") as fout:
        data = {
            "tstamp": round(finish_tstamp, 4),
            "task_type": "sts",
            "type": "chat",
            "model": model_name,
            "gen_params": {},
            "start": round(start_tstamp, 4),
            "finish": round(finish_tstamp, 4),
            "state": state.dict(),
            "ip": get_ip(request),
        }
        fout.write(json.dumps(data) + "\n")
        # append_json_item_on_log_server(data, get_conv_log_filename())


def check_input_sts(txt0, txt1, txt2):
    if any([not(x) for x in (txt0, txt1, txt2)]): raise gr.Warning("Prompt cannot be empty.")
    if len(set([txt0, txt1, txt2])) != 3: raise gr.Warning("Please input three different texts.")

def build_side_by_side_ui_anon_sts(models):
    notice_markdown = """
# ⚔️ MTEB Arena ⚔️ : STS 🤖
## 📜 Rules
- Input three different texts to two anonymous models and vote which visualizes their similarity better.
- Whenever you have decided which model is better, click the button below to vote.
- Click "New Round" to start a new round.

## 👇 Vote now!
"""
    state0 = gr.State()
    state1 = gr.State()
    gen_func = partial(sts_side_by_side, models.sts_parallel)
    
    gr.Markdown(notice_markdown, elem_id="notice_markdown")

    with gr.Group(elem_id="share-region-anon"):
        with gr.Accordion("🔍 Expand to see all Arena players", open=False):
            model_description_md = models.get_model_description_md()
            gr.Markdown(model_description_md, elem_id="model_description_markdown")
        with gr.Row():
            with gr.Column():
                chatbot_left = gr.HTML(label="Model A")
            with gr.Column():
                chatbot_right = gr.HTML(label="Model B")

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

    with gr.Row():
        clear_btn = gr.Button(value="🎲 New Round", interactive=False)

    gr.Markdown(acknowledgment_md, elem_id="ack_markdown")

    btn_list = [leftvote_btn, rightvote_btn, tie_btn, bothbad_btn, clear_btn]

    send_btn.click(
        check_input_sts,
        inputs=[textbox0, textbox1, textbox2],
        outputs=None
    ).success(
        partial(disable_buttons, 4),
        inputs=None,
        outputs=[send_btn, textbox0, textbox1, textbox2],   
    ).success(
        gen_func,
        inputs=[state0, state1, textbox0, textbox1, textbox2, model_selector_left, model_selector_right],
        outputs=[state0, state1, chatbot_left, chatbot_right, model_selector_left, model_selector_right],
        api_name="send_btn_anon"
    ).success(
        enable_buttons_side_by_side,
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
        partial(enable_buttons_side_by_side, 4),
        inputs=None,
        outputs=[send_btn, textbox0, textbox1, textbox2],
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
# ⚔️ MTEB Arena ⚔️ : STS 🤖
## 📜 Rules
- Input three different texts to two models and vote which visualizes their similarity better.
- Whenever you have decided which model is better, click the button below to vote.
- Click "New Round" to start a new round.

## 👇 Choose two models & vote now!
"""
    model_list = list(models.model_meta.keys())

    state0 = gr.State()
    state1 = gr.State()
    gen_func = partial(sts_side_by_side, models.sts_parallel)
    gr.Markdown(notice_markdown, elem_id="notice_markdown")

    with gr.Group(elem_id="share-region-named"):
        with gr.Row():
            with gr.Column():
                model_selector_left = gr.Dropdown(
                    choices=model_list,
                    value=model_list[0] if len(model_list) > 0 else "",
                    interactive=True,
                    show_label=False,
                    container=False,
                    allow_custom_value=True
                    )
            with gr.Column():
                model_selector_right = gr.Dropdown(
                    choices=model_list,
                    value=model_list[1] if len(model_list) > 1 else "",
                    interactive=True,
                    show_label=False,
                    container=False,
                    allow_custom_value=True
                    )
        with gr.Row():
            with gr.Accordion("🔍 Expand to see all model descriptions", open=False):
                model_description_md = models.get_model_description_md()
                gr.Markdown(model_description_md, elem_id="model_description_markdown")

        with gr.Row():
            with gr.Column():
                chatbot_left = gr.HTML(
                    label="Model A",
                )
            with gr.Column():
                chatbot_right = gr.HTML(
                    label="Model B",
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

    with gr.Row():
        clear_btn = gr.Button(value="🗑️  Clear history", interactive=False)

    gr.Markdown(acknowledgment_md, elem_id="ack_markdown")
    
    btn_list = [leftvote_btn, rightvote_btn, tie_btn, bothbad_btn, clear_btn]

    send_btn.click(
        check_input_sts,
        inputs=[textbox0, textbox1, textbox2],
        outputs=None
    ).success(
        partial(disable_buttons_side_by_side, 4),
        inputs=None,
        outputs=[send_btn, textbox0, textbox1, textbox2],   
    ).success(
        gen_func,
        inputs=[state0, state1, textbox0, textbox1, textbox2, model_selector_left, model_selector_right],
        outputs=[state0, state1, chatbot_left, chatbot_right],
        api_name="send_side_by_side"
    ).success(
        enable_buttons_side_by_side,
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
        partial(enable_buttons_side_by_side, 4),
        inputs=None,
        outputs=[send_btn, textbox0, textbox1, textbox2],
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
# MTEB Arena Playground: STS 🤖
"""
    #| [GitHub](https://github.com/embeddings-benchmark) |
    ## 🤖 Choose any model

    state = gr.State()
    gen_func = partial(sts, models.sts)
    gr.Markdown(notice_markdown, elem_id="notice_markdown")

    model_list = list(models.model_meta.keys())

    with gr.Row(elem_id="model_selector_row"):
        model_selector = gr.Dropdown(
            choices=model_list,
            value=model_list[0] if len(model_list) > 0 else "",
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

    with gr.Row():
        chatbot = gr.HTML(
            label="Model",
        )

    with gr.Row() as button_row:
        upvote_btn = gr.Button(value="👍  Upvote", interactive=False)
        downvote_btn = gr.Button(value="👎  Downvote", interactive=False)
        flag_btn = gr.Button(value="⚠️  Flag", interactive=False)
        clear_btn = gr.Button(value="🗑️  Clear history", interactive=False)

    gr.Markdown(acknowledgment_md, elem_id="ack_markdown")
    
    btn_list = [upvote_btn, downvote_btn, flag_btn, clear_btn]

    model_selector.change(
        clear_history_sts, 
        inputs=None, 
        outputs=[state, textbox0, textbox1, textbox2, chatbot], 
        api_name="model_selector_single"
    ).then(
        disable_buttons,
        inputs=None,
        outputs=btn_list
    ).then(
        partial(enable_buttons_side_by_side, 4),
        inputs=None,
        outputs=[send_btn, textbox0, textbox1, textbox2],
    )

    send_btn.click(
        check_input_sts,
        inputs=[textbox0, textbox1, textbox2],
        outputs=None
    ).success(
        partial(disable_buttons, 4),
        inputs=None,
        outputs=[send_btn, textbox0, textbox1, textbox2],   
    ).success(
        gen_func,
        inputs=[state, textbox0, textbox1, textbox2, model_selector],
        outputs=[state, chatbot],
        api_name="send_btn_single",
        show_progress="full"
    ).success(
        enable_buttons,
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
        disable_buttons,
        inputs=None,
        outputs=btn_list
    ).then(
        partial(enable_buttons_side_by_side, 4),
        inputs=None,
        outputs=[send_btn, textbox0, textbox1, textbox2],
    )



