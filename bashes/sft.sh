#!/bin/bash
#SBATCH --account=rrg-dsuth
#SBATCH --gres=gpu:v100l:1              # Number of GPU(s) per node
#SBATCH --cpus-per-task=4         # CPU cores/threads
#SBATCH --mem=32000M               # memory per node
#SBATCH --time=2-10:00            # time (DD-HH:MM)
#SBATCH --output=./logs/stage1.txt 
#SBATCH --job-name=pythia1b

# 1. Load the required modules
module load python/3.10 StdEnv/2023 cudacore/.12.2.2 arrow/14.0.0

# 2. Load your environment
source /home/joshua52/projects/def-dsuth/joshua52/env_llm/bin/activate

# 3. Go to the correct path
cd /home/joshua52/projects/def-dsuth/joshua52/finetuning_dynamics

python -u train.py model=qwen exp_name=sft_qwen05_ep10 trainer=BasicTrainer n_epochs=10 n_examples=50000
#python -u train.py model=pythia14 exp_name=pythia14_supreject_20240419 trainer=BasicTrainer n_epochs=4 train_supervise=rejected
#python -u train.py model=pythia410m exp_name=sft_pythia410m_save_ep4 trainer=BasicTrainer n_epochs=4 n_examples=20000
#python -u train.py model=pythia1b exp_name=sft_pythia1b_save_ep4 trainer=BasicTrainer n_epochs=2 n_examples=10000