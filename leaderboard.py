import pickle

import gradio as gr
import numpy as np
import pandas as pd

from arena_elo.elo_analysis import load_results

leader_component_values = [None] * 5

TASK_TYPE_TO_EMOJI = {
    "Retrieval": "🔎",
    "Clustering": "✨",
    "STS": "☘️",
}

def make_arena_leaderboard_md(elo_results):
    arena_df = elo_results["leaderboard_table_df"]
    last_updated = elo_results["last_updated_datetime"]
    total_votes = sum(arena_df["num_battles"]) // 2
    total_models = len(arena_df)

    leaderboard_md = f"""
Total #models: **{total_models}**.&nbsp;&nbsp;&nbsp;&nbsp;Total #votes: **{total_votes}**.&nbsp;&nbsp;&nbsp;&nbsp;Last updated: {last_updated}.

Contribute your votes 🗳️ at [MTEB Arena](https://huggingface.co/spaces/mteb/arena)! 
"""
    return leaderboard_md

def model_hyperlink(model_name, link):
    return f'<a target="_blank" href="{link}" style="color: var(--link-text-color); text-decoration: underline;text-decoration-style: dotted;">{model_name}</a>'

def load_leaderboard_table_csv(filename, add_hyperlink=True):
    df = pd.read_csv(filename)
    for col in df.columns:
        if "Arena Elo rating" in col:
            df[col] = df[col].apply(lambda x: int(x) if x != "-" else np.nan)
        elif col in ("MTEB Overall Avg", "MTEB Retrieval Avg", "MTEB Clustering Avg", "MTEB STS Avg"):
            df[col] = df[col].apply(lambda x: x if x != "-" else np.nan)
        if add_hyperlink and col == "Model":
            df[col] = df.apply(lambda row: model_hyperlink(row[col], row["Link"]), axis=1)
    return df

def get_arena_table(arena_df, model_table_df, task_type="Retrieval"):
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
        row.append(model_table_df.iloc[i]["MTEB Overall Avg"])
        row.append(model_table_df.iloc[i][f"MTEB {task_type} Avg"])
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

def build_leaderboard_tab(elo_results_file, leaderboard_table_file, show_plot=False, task_type="Retrieval"):
    if elo_results_file is None:  # Do live update
        md = "Loading ..."
        p1 = p2 = p3 = p4 = None
    else:
        elo_results = load_results(elo_results_file)
        anony_elo_results = elo_results["anony"]
        anony_arena_df = anony_elo_results["leaderboard_table_df"]
        p1 = anony_elo_results["win_fraction_heatmap"]
        p2 = anony_elo_results["battle_count_heatmap"]
        p3 = anony_elo_results["bootstrap_elo_rating"]
        p4 = anony_elo_results["average_win_rate_bar"]

        md = f"""
# 🏆 MTEB Arena Leaderboard: {task_type} {TASK_TYPE_TO_EMOJI[task_type]}
"""
    # | [GitHub](https://github.com/embeddings-benchmark) |
    md_1 = gr.Markdown(md, elem_id="leaderboard_markdown")

    if leaderboard_table_file:
        model_table_df = load_leaderboard_table_csv(leaderboard_table_file)
        arena_table_vals = get_arena_table(anony_arena_df, model_table_df, task_type=task_type)
        md = make_arena_leaderboard_md(anony_elo_results)
        gr.Markdown(md, elem_id="leaderboard_markdown")
        gr.Dataframe(
            headers=[
                "Rank",
                "🤖 Model",
                "⭐ MTEB Arena Elo",
                "📊 95% CI",
                "🗳️ Votes",
                "🥇 MTEB Overall Avg",
                f"🥇 MTEB {task_type} Avg",                        
                "Organization",
                "License",
            ],
            datatype=[
                "str",
                "markdown",
                "number",
                "str",
                "number",
                "number", 
                "number",
                "str",
                "str",
            ],
            value=arena_table_vals,
            elem_id="arena_leaderboard_dataframe",
            height=700,
            column_widths=[50, 150, 100, 100, 100, 100, 100, 150, 150],
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
    # return [md_1, plot_1, plot_2, plot_3, plot_4]
    return [md_1]