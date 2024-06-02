# set LOGDIR to default if not set before
if [ -z "$LOGDIR" ]; then
    echo "LOGDIR is not set. Using default './MTEB-Arena-logs/vote_log'"
    export LOGDIR="./MTEB-Arena-logs/vote_log"
fi

mkdir -p results

# For battle data
python -m arena_elo.clean_battle_data --task_name "retrieval"
retrieval_battle_cutoff_date=`cat cut_off_date.txt` && rm cut_off_date.txt && echo "Retrieval battle data last updated on $retrieval_battle_cutoff_date"

mkdir -p ./results/$retrieval_battle_cutoff_date

cp clean_battle_retrieval_$retrieval_battle_cutoff_date.json ./results/latest/clean_battle_retrieval.json
mv clean_battle_retrieval_$retrieval_battle_cutoff_date.json ./results/$retrieval_battle_cutoff_date/clean_battle_retrieval.json

python3 -m arena_elo.elo_analysis --clean-battle-file ./results/$retrieval_battle_cutoff_date/clean_battle_retrieval.json --num-bootstrap 1
mv ./elo_results_$retrieval_battle_cutoff_date.pkl ./results/$retrieval_battle_cutoff_date/elo_results_retrieval.pkl

python -m arena_elo.generate_leaderboard \
    --elo_rating_pkl "./results/$retrieval_battle_cutoff_date/elo_results_retrieval.pkl" \
    --output_csv "./results/$retrieval_battle_cutoff_date/retrieval_leaderboard.csv"

mkdir -p ./results/latest
cp ./results/$retrieval_battle_cutoff_date/retrieval_leaderboard.csv ./results/latest/retrieval_leaderboard.csv
cp ./results/$retrieval_battle_cutoff_date/elo_results_retrieval.pkl ./results/latest/elo_results_retrieval.pkl
