#!/bin/bash -l
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=5:00:00
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --account=plgbcfg-gpu-gh200
#SBATCH --partition=plgrid-gpu-gh200
#SBATCH --output=act-out-%j.log
#SBATCH --error=act-err-%j.log

# IMPORTANT: load the modules for machine learning tasks and libraries
ml ML-bundle/25.10

cd $SCRATCH/TopKSAE-research

unset PIP_EXTRA_INDEX_URL
export HF_HOME=$SCRATCH/hf_home

source env/bin/activate

python precompute_activations.py -d cc3m -m ViT-B~16 -s -w 1
