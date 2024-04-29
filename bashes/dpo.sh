#!/bin/bash
#SBATCH --account=rrg-dsuth
#SBATCH --gres=gpu:v100l:1              # Number of GPU(s) per node
#SBATCH --cpus-per-task=4         # CPU cores/threads
#SBATCH --mem=32000M               # memory per node
#SBATCH --time=2-00:00            # time (DD-HH:MM)
#SBATCH --output=./logs/stage1.txt 
#SBATCH --job-name=dpo

# 1. Load the required modules
module load python/3.10 StdEnv/2023 cudacore/.12.2.2 arrow/14.0.0

# 2. Load your environment
source /home/joshua52/projects/def-dsuth/joshua52/env_llm/bin/activate

# 3. Go to the correct path
cd /home/joshua52/projects/def-dsuth/joshua52/finetuning_dynamics

# -------- One GPU is fine
#python -u train.py loss=dpo loss.beta=0.1 pre_sft_steps=2000 model=pythia410m exp_name=dpo_pythia410m_sft2000 trainer=BasicTrainer n_epochs=4
#python -u train.py loss=dpo loss.beta=0.1 pre_sft_steps=0 model=pythia410m exp_name=dpo_pythia410m_sft0 trainer=BasicTrainer n_epochs=4

#python -u train.py loss=dpo loss.beta=0.1 pre_sft_steps=1000 model=pythia1b exp_name=dpo_pythia1b_sft1000 trainer=BasicTrainer n_epochs=4
#python -u train.py loss=dpo loss.beta=0.1 pre_sft_steps=0 model=pythia1b exp_name=dpo_pythia1b_sft0 trainer=BasicTrainer n_epochs=4
#python -u train.py loss=dpo loss.beta=0.1 pre_sft_steps=1 model=pythia14 exp_name=dpo_pythia14 trainer=BasicTrainer n_epochs=4
python -u train.py loss=dpo loss.beta=0.1 model=qwen exp_name=dpo_qwen05_sft10 model.archive=sft_qwen05_ep10 trainer=BasicTrainer n_epochs=8 eval_batch_size=2 n_examples=40000

#python -u train.py loss=dpo loss.beta=0.1 model=pythia410m exp_name=dpo_pythia410m_sft_ep10 model.archive=sft_pythia410m_save trainer=BasicTrainer n_epochs=4 n_examples=null

# -------- Need multi GPU
#python -u train.py loss=dpo loss.beta=0.1 model=pythia28 exp_name=dpo_pythia28 trainer=BasicTrainer n_epochs=4 n_examples=20000 eval_batch_size=20