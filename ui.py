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

retrieval_logger = build_logger("gradio_retrieval", "gradio_retrieval.log") # logger for side-by-side and battle

disable_btn = gr.update(interactive=False)

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

def clear_history_side_by_side():
    return None, None, "", None, None

def clear_history_side_by_side_anon():
    return None, None, "", None, None, gr.Markdown("", visible=False), gr.Markdown("", visible=False)

def enable_buttons(i=5):
    return tuple(gr.update(interactive=True) for _ in range(i))

def disable_buttons(i=5):
    return tuple(gr.update(interactive=False) for _ in range(i))

def disable_buttons_side_by_side(i=6):
    return tuple(gr.update(visible=i>=4, interactive=False) for i in range(i))

def enable_buttons_side_by_side(i=6):
    return tuple(gr.update(visible=True, interactive=True) for i in range(i))

def vote_last_response(vote_type, state0, state1, model_selector0, model_selector1, request: gr.Request):
    retrieval_logger.info(f"{vote_type} (named). ip: {get_ip(request)}")
    with open(get_conv_log_filename(), "a") as fout:
        data = {
            "tstamp": round(time.time(), 4),
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


def vote_last_response_single(vote_type, state, model_selector, request: gr.Request):
    retrieval_logger.info(f"{vote_type} (named). ip: {get_ip(request)}")
    with open(get_conv_log_filename(), "a") as fout:
        data = {
            "tstamp": round(time.time(), 4),
            "type": vote_type,
            "models": model_selector,
            "states": state.dict(),
            "ip": get_ip(request),
        }
        fout.write(json.dumps(data) + "\n")
    return ("",) + (disable_btn,) * 3


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
    
def retrieve_anon(gen_func, state0, state1, text, model_name0, model_name1, request: gr.Request):
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

def build_side_by_side_ui_anon(models):
    notice_markdown = """
# ⚔️  MTEB Arena ⚔️ : Massive Text Embedding Benchmark Arena
## 📜 Rules
- Input a search query to two anonymous models and vote which retrieves the better passage for your query.
- You can search multiple times with the same two anonymous models, but they will not use the past result to search.
- Whenever you have decided which model is better, click the button below to vote.
- Click "New Round" to start a new round.

## 👇 Vote now!
"""

    state0 = gr.State()
    state1 = gr.State()
    gen_func = partial(retrieve_anon, models.retrieve_parallel_anon)
    
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
            slow_warning = gr.Markdown("", elem_id="notice_markdown")

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
        inputs=None,
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
# ⚔️  MTEB Arena ⚔️ : Massive Text Embedding Benchmark Arena

## 📜 Rules
- Input a search query to two anonymous models and vote which retrieves the better passage for your query.
- You can search multiple times with the same two anonymous models, but they will not use the past result to search.
- Whenever you have decided which model is better, click the button below to vote.
- Click "New Round" to start a new round.

## 👇 Choose two models & vote now!
"""
    model_list = list(models.model_meta.keys())

    state0 = gr.State()
    state1 = gr.State()
    gen_func = partial(retrieve_anon, models.retrieve_parallel_anon)
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
# 🏔️ Play with Retrieval Models
| [GitHub](https://github.com/embeddings-benchmark) | 

## 🤖 Choose any retriever
"""

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

    model_selector.change(clear_history, inputs=None, outputs=[state, textbox, chatbot], api_name="model_selector_single")
    
    btn_list = [upvote_btn, downvote_btn, flag_btn, clear_btn]

    textbox.submit(
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

