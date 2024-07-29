"""Creates slurm jobs for running models on all tasks"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Iterable

from yaml import safe_load

def create_slurm_job_file(
    model_name: str,
    corpus: str,
    slurm_prefix: str,
    slurm_jobs_folder: Path,
) -> Path:
    """Create slurm job file for running a model on a task"""
    slurm_job = f"{slurm_prefix}\n"
    slurm_job += f"python run_vertex_index.py {model_name} {corpus}"

    model_path_name = model_name.replace("/", "__")

    slurm_job_file = slurm_jobs_folder / f"{model_path_name}_{corpus}.sh"
    with open(slurm_job_file, "w") as f:
        f.write(slurm_job)
    return slurm_job_file


def create_slurm_job_files(
    model_names: list[str],
    corpora: Iterable[mteb.AbsTask],
    slurm_prefix: str,
    slurm_jobs_folder: Path,
) -> list[Path]:
    """Create slurm job files for running models on all tasks"""
    slurm_job_files = []
    for model_name in model_names:
        for corpus in corpora:
            slurm_job_file = create_slurm_job_file(
                model_name,
                corpus,
                slurm_prefix,
                slurm_jobs_folder,
            )
            slurm_job_files.append(slurm_job_file)
    return slurm_job_files


def run_slurm_jobs(files: list[Path]) -> None:
    """Run slurm jobs based on the files provided"""
    for file in files:
        subprocess.run(["sbatch", file])


if __name__ == "__main__":
    # SHOULD BE UPDATED
    slurm_prefix = """#!/bin/bash
#SBATCH --job-name=mtebarena
#SBATCH --nodes=1
#SBATCH --partition=a3mixed
#SBATCH --gres=gpu:1                 # number of gpus
#SBATCH --time 24:00:00             # maximum execution time (HH:MM:SS)
#SBATCH --output=/data/niklas/jobs/%x-%j.out           # output file name
#SBATCH --exclusive
"""

    MODEL_META_PATH = "model_meta.yml"
    with open(MODEL_META_PATH, 'r', encoding='utf-8') as f:
        model_meta = safe_load(f)

    model_names = model_meta["model_meta"].keys()
    models_to_remove = ["nvidia/NV-Embed-v1"]
    model_names = [model for model in model_names if model not in models_to_remove]

    corpora = ["wikipedia", "arxiv", "stackexchange"]
    slurm_jobs_folder = Path("slurm_jobs")

    slurm_jobs_folder.mkdir(exist_ok=True)
    files = create_slurm_job_files(
        model_names, corpora, slurm_prefix, slurm_jobs_folder
    )
    run_slurm_jobs(files)
