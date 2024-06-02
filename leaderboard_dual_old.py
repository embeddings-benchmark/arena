import pickle

import gradio as gr
import numpy as np
import pandas as pd


acknowledgment_md = """
### Acknowledgment
We thank X, Y, Z for their generous sponsorship. If you would like to sponsor us, please get in touch.

We thank [Chatbot Arena](https://chat.lmsys.org/), [Vision Arena](https://huggingface.co/spaces/WildVision/vision-arena) and [GenAI-Arena](https://huggingface.co/spaces/TIGER-Lab/GenAI-Arena) for their great work.
"""

leader_component_values = [None] * 5

def make_full_leaderboard_md(elo_results):
    arena_df = elo_results["leaderboard_table_df"]
    last_updated = elo_results["last_updated_datetime"]
    total_votes = sum(arena_df["num_battles"]) // 2
    total_models = len(arena_df)

    leaderboard_md = f"""
Total #models: **{total_models}**(full:anonymous+open). Total #votes: **{total_votes}**. Last updated: {last_updated}.

Contribute your vote 🗳️ at [MTEB Arena](https://huggingface.co/spaces/mteb/arena)! 
"""
    return leaderboard_md

def make_arena_leaderboard_md(elo_results):
    arena_df = elo_results["leaderboard_table_df"]
    last_updated = elo_results["last_updated_datetime"]
    total_votes = sum(arena_df["num_battles"]) // 2
    total_models = len(arena_df)

    leaderboard_md = f"""
Total #models: **{total_models}**(anonymous). Total #votes: **{total_votes}**. Last updated: {last_updated}.
(Note: Only anonymous votes are considered here. Check the full leaderboard for all votes.)

Contribute the votes 🗳️ at [GenAI-Arena](https://huggingface.co/spaces/TIGER-Lab/GenAI-Arena)! 

If you want to see more models, please help us [add them](https://github.com/TIGER-AI-Lab/ImagenHub?tab=readme-ov-file#-contributing-).
"""
    return leaderboard_md

def get_full_table(anony_arena_df, full_arena_df, model_table_df):
    values = []
    for i in range(len(model_table_df)):
        row = []
        model_key = model_table_df.iloc[i]["key"]
        model_name = model_table_df.iloc[i]["Model"]
        # model display name
        row.append(model_name)
        if model_key in anony_arena_df.index:
            idx = anony_arena_df.index.get_loc(model_key)
            row.append(round(anony_arena_df.iloc[idx]["rating"]))
        else:
            row.append(np.nan)
        if model_key in full_arena_df.index:
            idx = full_arena_df.index.get_loc(model_key)
            row.append(round(full_arena_df.iloc[idx]["rating"]))
        else:
            row.append(np.nan)
        # row.append(model_table_df.iloc[i]["MT-bench (score)"])
        # row.append(model_table_df.iloc[i]["Num Battles"])
        row.append(model_table_df.iloc[i]["MTEB Overall Avg"])
        row.append(model_table_df.iloc[i]["MTEB Retrieval Avg"])
        # Organization
        row.append(model_table_df.iloc[i]["Organization"])
        # license
        row.append(model_table_df.iloc[i]["License"])

        values.append(row)
    values.sort(key=lambda x: -x[1] if not np.isnan(x[1]) else 1e9)
    return values

def model_hyperlink(model_name, link):
    return f'<a target="_blank" href="{link}" style="color: var(--link-text-color); text-decoration: underline;text-decoration-style: dotted;">{model_name}</a>'

def load_leaderboard_table_csv(filename, add_hyperlink=True):
    df = pd.read_csv(filename)
    for col in df.columns:
        if "Arena Elo rating" in col:
            df[col] = df[col].apply(lambda x: int(x) if x != "-" else np.nan)
        elif col in ("MTEB Overall Avg", "MTEB Retrieval Avg"):
            df[col] = df[col].apply(lambda x: x if x != "-" else np.nan)
        if add_hyperlink and col == "Model":
            df[col] = df.apply(lambda row: model_hyperlink(row[col], row["Link"]), axis=1)
    return df

def get_arena_table(arena_df, model_table_df):
    # sort by rating
    arena_df = arena_df.sort_values(by=["rating"], ascending=False)
    values = []
    for i in range(len(arena_df)):
        row = []
        model_key = arena_df.index[i]
        model_name = model_table_df[model_table_df["key"] == model_key]["Model"].values[0]
        # rank
        row.append(i + 1)
        # model display name
        row.append(model_name)
        # elo rating
        row.append(round(arena_df.iloc[i]["rating"]))
        upper_diff = round(arena_df.iloc[i]["rating_q975"] - arena_df.iloc[i]["rating"])
        lower_diff = round(arena_df.iloc[i]["rating"] - arena_df.iloc[i]["rating_q025"])
        row.append(f"+{upper_diff}/-{lower_diff}")
        # num battles
        row.append(round(arena_df.iloc[i]["num_battles"]))
        # Organization
        row.append(
            model_table_df[model_table_df["key"] == model_key]["Organization"].values[0]
        )
        # license
        row.append(
            model_table_df[model_table_df["key"] == model_key]["License"].values[0]
        )
        values.append(row)
    return values

def build_leaderboard_tab(elo_results_file, leaderboard_table_file, show_plot=False):
    if elo_results_file is None:  # Do live update
        md = "Loading ..."
        p1 = p2 = p3 = p4 = None
    else:
        with open(elo_results_file, "rb") as fin:
            elo_results = pickle.load(fin)

        anony_elo_results = elo_results["anony"]
        full_elo_results = elo_results["full"]
        anony_arena_df = anony_elo_results["leaderboard_table_df"]
        full_arena_df = full_elo_results["leaderboard_table_df"]
        p1 = anony_elo_results["win_fraction_heatmap"]
        p2 = anony_elo_results["battle_count_heatmap"]
        p3 = anony_elo_results["bootstrap_elo_rating"]
        p4 = anony_elo_results["average_win_rate_bar"]

        md = f"""
# 🏆 MTEB Arena Leaderboard
| [GitHub](https://github.com/embeddings-benchmark) |

"""

    md_1 = gr.Markdown(md, elem_id="leaderboard_markdown")

    if leaderboard_table_file:
        model_table_df = load_leaderboard_table_csv(leaderboard_table_file)
        with gr.Tabs() as tabs:
            # arena table
            arena_table_vals = get_arena_table(anony_arena_df, model_table_df)
            with gr.Tab("Arena Elo", id=0):
                md = make_arena_leaderboard_md(anony_elo_results)
                gr.Markdown(md, elem_id="leaderboard_markdown")
                gr.Dataframe(
                    headers=[
                        "Rank",
                        "🤖 Model",
                        "⭐ MTEB Arena Elo",
                        "📊 95% CI",
                        "🗳️ Votes",
                        "Organization",
                        "License",
                    ],
                    datatype=[
                        "str",
                        "markdown",
                        "number",
                        "str",
                        "number",
                        "str",
                        "str",
                    ],
                    value=arena_table_vals,
                    elem_id="arena_leaderboard_dataframe",
                    height=700,
                    column_widths=[50, 200, 100, 100, 100, 150, 150],
                    wrap=True,
                )
            with gr.Tab("Full Leaderboard", id=1):
                md = make_full_leaderboard_md(full_elo_results)
                gr.Markdown(md, elem_id="leaderboard_markdown")
                full_table_vals = get_full_table(anony_arena_df, full_arena_df, model_table_df)
                gr.Dataframe(
                    headers=[
                        "🤖 Model",
                        "⭐ MTEB Arena Elo (anony)",
                        "⭐ MTEB Arena Elo (full)",
                        "🥇 MTEB Overall Avg",
                        "🥇 MTEB Retrieval Avg",
                        "Organization",
                        "License",
                    ],
                    datatype=["markdown", "number", "number", "number", "number", "str", "str"],
                    value=full_table_vals,
                    elem_id="full_leaderboard_dataframe",
                    column_widths=[200, 100, 100, 100, 100, 100, 150, 150],
                    height=700,
                    wrap=True,
                )
        if not show_plot:
            gr.Markdown(
                """## We are still collecting more votes on more models. The ranking will be updated very frequently. Please stay tuned!""",
                elem_id="leaderboard_markdown",
            )
    else:
        pass

    leader_component_values[:] = [md, p1, p2, p3, p4]

    """
    with gr.Row():
        with gr.Column():
            gr.Markdown(
                "#### Figure 1: Fraction of Model A Wins for All Non-tied A vs. B Battles"
            )
            plot_1 = gr.Plot(p1, show_label=False)
        with gr.Column():
            gr.Markdown(
                "#### Figure 2: Battle Count for Each Combination of Models (without Ties)"
            )
            plot_2 = gr.Plot(p2, show_label=False)
    with gr.Row():
        with gr.Column():
            gr.Markdown(
                "#### Figure 3: Bootstrap of Elo Estimates (1000 Rounds of Random Sampling)"
            )
            plot_3 = gr.Plot(p3, show_label=False)
        with gr.Column():
            gr.Markdown(
                "#### Figure 4: Average Win Rate Against All Other Models (Assuming Uniform Sampling and No Ties)"
            )
            plot_4 = gr.Plot(p4, show_label=False)
    """

    gr.Markdown(acknowledgment_md)

    # return [md_1, plot_1, plot_2, plot_3, plot_4]
    return [md_1]