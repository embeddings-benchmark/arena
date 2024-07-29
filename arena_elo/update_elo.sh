#!/bin/bash

mkdir -p results

# For battle data
for task in "retrieval" "clustering" "sts"; do
    python -m arena_elo.clean_battle_data --task_name $task
    battle_cutoff_date=`cat cut_off_date.txt` && rm cut_off_date.txt && echo "$task battle data last updated on $battle_cutoff_date"
    mkdir -p ./results/$battle_cutoff_date
    cp clean_battle_${task}_$battle_cutoff_date.json ./results/latest/clean_battle_$task.json
    mv clean_battle_${task}_$battle_cutoff_date.json ./results/$battle_cutoff_date/clean_results_${task}.json
    python3 -m arena_elo.elo_analysis --clean-battle-file ./results/$battle_cutoff_date/clean_results_${task}.json
    mv ./elo_results_$battle_cutoff_date.pkl ./results/$battle_cutoff_date/elo_results_${task}.pkl
    python -m arena_elo.generate_leaderboard \
        --elo_rating_pkl "./results/$battle_cutoff_date/elo_results_${task}.pkl" \
        --output_csv "./results/$battle_cutoff_date/${task}_leaderboard.csv"
    mkdir -p ./results/latest
    cp ./results/$battle_cutoff_date/${task}_leaderboard.csv ./results/latest/${task}_leaderboard.csv
    cp ./results/$battle_cutoff_date/elo_results_${task}.pkl ./results/latest/elo_results_${task}.pkl
    echo "$task leaderboard updated"
done

