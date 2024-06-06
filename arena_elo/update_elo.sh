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


echo "Retrieval leaderboard updated"


python -m arena_elo.clean_battle_data --task_name "clustering"
clustering_battle_cutoff_date=`cat cut_off_date.txt` && rm cut_off_date.txt && echo "Clustering battle data last updated on $clustering_battle_cutoff_date"
mkdir -p ./results/$clustering_battle_cutoff_date
cp clean_battle_clustering_$clustering_battle_cutoff_date.json ./results/latest/clean_battle_clustering.json
mv clean_battle_clustering_$clustering_battle_cutoff_date.json ./results/$clustering_battle_cutoff_date/clean_battle_clustering.json
python3 -m arena_elo.elo_analysis --clean-battle-file ./results/$clustering_battle_cutoff_date/clean_battle_clustering.json --num-bootstrap 1
mv ./elo_results_$clustering_battle_cutoff_date.pkl ./results/$clustering_battle_cutoff_date/elo_results_clustering.pkl
python -m arena_elo.generate_leaderboard \
    --elo_rating_pkl "./results/$clustering_battle_cutoff_date/elo_results_clustering.pkl" \
    --output_csv "./results/$clustering_battle_cutoff_date/clustering_leaderboard.csv"
mkdir -p ./results/latest
cp ./results/$clustering_battle_cutoff_date/clustering_leaderboard.csv ./results/latest/clustering_leaderboard.csv
cp ./results/$clustering_battle_cutoff_date/elo_results_clustering.pkl ./results/latest/elo_results_clustering.pkl

echo "Clustering leaderboard updated"


python -m arena_elo.clean_battle_data --task_name "sts"
sts_battle_cutoff_date=`cat cut_off_date.txt` && rm cut_off_date.txt && echo "STS battle data last updated on $sts_battle_cutoff_date"
mkdir -p ./results/$sts_battle_cutoff_date
cp clean_battle_sts_$sts_battle_cutoff_date.json ./results/latest/clean_battle_sts.json
mv clean_battle_sts_$sts_battle_cutoff_date.json ./results/$sts_battle_cutoff_date/clean_battle_sts.json
python3 -m arena_elo.elo_analysis --clean-battle-file ./results/$sts_battle_cutoff_date/clean_battle_sts.json --num-bootstrap 1
mv ./elo_results_$sts_battle_cutoff_date.pkl ./results/$sts_battle_cutoff_date/elo_results_sts.pkl
python -m arena_elo.generate_leaderboard \
    --elo_rating_pkl "./results/$sts_battle_cutoff_date/elo_results_sts.pkl" \
    --output_csv "./results/$sts_battle_cutoff_date/sts_leaderboard.csv"
mkdir -p ./results/latest
cp ./results/$sts_battle_cutoff_date/sts_leaderboard.csv ./results/latest/sts_leaderboard.csv
cp ./results/$sts_battle_cutoff_date/elo_results_sts.pkl ./results/latest/elo_results_sts.pkl

echo "STS leaderboard updated"
