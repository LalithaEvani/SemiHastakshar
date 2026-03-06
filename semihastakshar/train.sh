#!/bin/bash
#SBATCH -A xxx
#SBATCH --nodelist=gnode0xx
#SBATCH -c 18
#SBATCH --gres=gpu:2
#SBATCH --mem-per-cpu=2G
#SBATCH --time=4-00:00:00
#SBATCH --output=bash_outputs/output.txt

conda activate my_env

cd /path/to/code

python run_pseudo_labeling.py
