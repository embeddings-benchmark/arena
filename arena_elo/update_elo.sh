#!/bin/bash

mkdir -p results
mkdir -p results/latest
# For battle data

for task in "retrieval" "clustering" "sts"; do
    python -m arena_elo.clean_battle_data --task_name $task
    battle_cutoff_date=`cat cut_off_date.txt` && rm cut_off_date.txt && echo "$task battle data last updated on $battle_cutoff_date"
    mkdir -p ./results/$battle_cutoff_date
    cp clean_battle_${task}_$battle_cutoff_date.json ./results/latest/clean_battle_$task.json
    mv clean_battle_${task}_$battle_cutoff_date.json ./results/$battle_cutoff_date/clean_results_${task}.json
    python3 -m arena_elo.elo_analysis --clean-battle-file ./results/$battle_cutoff_date/clean_results_${task}.json --num-bootstrap 1
    mv ./elo_results_$battle_cutoff_date ./results/$battle_cutoff_date/elo_results_${task}
    cmd="""python -m arena_elo.generate_leaderboard \
        --elo_rating_folder "./results/$battle_cutoff_date/elo_results_${task}" \
        --output_csv "./results/$battle_cutoff_date/${task}_leaderboard.csv""""
    echo $cmd
    eval $cmd
    mkdir -p ./results/latest
    cp ./results/$battle_cutoff_date/${task}_leaderboard.csv ./results/latest/${task}_leaderboard.csv
    cp -R ./results/$battle_cutoff_date/elo_results_${task} ./results/latest/elo_results_${task}
    echo "$task leaderboard updated"
done

