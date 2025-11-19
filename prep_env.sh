#!/bin/bash -l
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=00:15:00
#SBATCH --account=plgbcfg-gpu-gh200
#SBATCH --partition=plgrid-gpu-gh200
#SBATCH --output=prep-out-%j.log
#SBATCH --error=prep-err-%j.log

# IMPORTANT: load the modules for machine learning tasks and libraries
ml ML-bundle/25.10

cd $SCRATCH/TopKSAE-research

unset PIP_EXTRA_INDEX_URL
export HF_HOME=$SCRATCH/hf_home

# create and activate the virtual environment 
python -m venv  env/
source env/bin/activate

# install torch + torchvisionn with CUDA support
pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cu129

# install the rest of requirements, for via requirements file
pip install --no-cache-dir -r requirements.txt