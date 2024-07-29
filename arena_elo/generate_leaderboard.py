import fire
import json
import pandas as pd
import pickle
from yaml import safe_load

RENAME_KEYS = {
    "organization": "Organization",
    "license": "License",
    "link": "Link",
    "mteb_overall": "MTEB Overall Avg",
    "mteb_retrieval": "MTEB Retrieval Avg",
    "mteb_clustering": "MTEB Clustering Avg",
    "mteb_sts": "MTEB STS Avg"
}

def main(
    elo_rating_pkl: str,
    output_csv: str
):    

    MODEL_META_PATH = "model_meta.yml"
    # Debugging
    # MODEL_META_PATH = "model_meta_debug.yml"
    with open(MODEL_META_PATH, 'r', encoding='utf-8') as f:
        model_info = safe_load(f)["model_meta"]

    # Rename keys
    for key in RENAME_KEYS:
        for model in model_info:
            if key in model_info[model]:
                model_info[model][RENAME_KEYS[key]] = model_info[model].pop(key)

    with open(elo_rating_pkl, "rb") as fin:
        elo_rating_results = pickle.load(fin)

    anony_elo_rating_results = elo_rating_results["anony"]
    full_elo_rating_results = elo_rating_results["full"]
    anony_leaderboard_data = anony_elo_rating_results["leaderboard_table_df"]
    full_leaderboard_data = full_elo_rating_results["leaderboard_table_df"]

    # Model,MT-bench (score),Arena Elo rating,MMLU,License,Link
    fields = ["key", "Model", "Arena Elo rating (anony)", "Arena Elo rating (full)", "MTEB Overall Avg", "MTEB Retrieval Avg", "MTEB Clustering Avg", "MTEB STS Avg", "License", "Organization", "Link"]
    # set Organization and license to empty for now
    all_models = anony_leaderboard_data.index.tolist()

    for model in all_models:
        if not model in model_info:
            model_info[model] = {}
            model_info[model]["MTEB Overall Avg"] = "N/A"
            model_info[model]["MTEB Retrieval Avg"] = "N/A"
            model_info[model]["MTEB Clustering Avg"] = "N/A"
            model_info[model]["MTEB STS Avg"] = "N/A"
            model_info[model]["License"] = "N/A"
            model_info[model]["Organization"] = "N/A"
            model_info[model]["Link"] = "N/A"
            print(f"Model {model} not found in model_info.json")
        model_info[model]["Model"] = model
        model_info[model]["key"] = model

        if model in anony_leaderboard_data.index:
            model_info[model]["Arena Elo rating (anony)"] = anony_leaderboard_data.loc[model, "rating"]
        else:
            model_info[model]["Arena Elo rating (anony)"] = 0

        if model in full_elo_rating_results["leaderboard_table_df"].index:
            model_info[model]["Arena Elo rating (full)"] = full_leaderboard_data.loc[model, "rating"]
        else:
            model_info[model]["Arena Elo rating (full)"] = 0
        # if model in anony_leaderboard_data.index:
        #     model_info[model]["Arena Elo rating"] = anony_leaderboard_data.loc[model, "rating"]
        # else:
        #     model_info[model]["Arena Elo rating"] = 0

    final_model_info = {}
    for model in model_info:
        if "Model" in model_info[model]:
            final_model_info[model] = model_info[model]
    model_info = final_model_info

    exclude_keys = ['starting_from']
    for key in exclude_keys:
        for model in model_info:
            if key in model_info[model]:
                del model_info[model][key]
    df = pd.DataFrame(model_info).T
    df = df[fields]
    # sort by anony rating
    df = df.sort_values(by=["Arena Elo rating (anony)"], ascending=False)
    df.to_csv(output_csv, index=False)
    print("Leaderboard data saved to", output_csv)
    print(df)


if __name__ == "__main__":
    fire.Fire(main)