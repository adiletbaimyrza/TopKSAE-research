#!/bin/bash -l
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=00:20:00
#SBATCH --account=plgbcfg-gpu-gh200
#SBATCH --partition=plgrid-gpu-gh200
#SBATCH --output=prep-out-%j.log
#SBATCH --error=prep-err-%j.log

# IMPORTANT: load the modules for machine learning tasks and libraries
ml ML-bundle/24.06a

cd $SCRATCH

unset PIP_EXTRA_INDEX_URL

# create and activate the virtual environment 
python -m venv  env/
source env/bin/activate

# install one of torch versions available at Helios wheel repo
pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cu129

# install the rest of requirements, for example via requirements file
pip install --no-cache-dir -r requirements.txt