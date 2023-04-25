#!/bin/bash -l

#SBATCH --job-name=attack-val-dl
#SBATCH --partition=gpu,gpub
#SBATCH --time=10:20:00
#SBATCH --begin=now
#SBATCH --gres=gpu:1080:1
#SBATCH --mem=20000M		    # was 50000/20000/15000
#SBATCH --cpus-per-task=4       # 2 or 4
#SBATCH --exclude=gpu06,gpu05

# Replace with your dataset or checkpoint path
export IFN_DIR_DATASET=/beegfs/data/shared
export IFN_DIR_CHECKPOINT="${PWD}/../../../experiments_code-release/"

# Replace with your own system path
export PYTHONPATH=/beegfs/work/kusuma/papers/cvpr2023/code_release/
export PYTHONPATH="${PYTHONPATH}:/beegfs/work/kusuma/papers/cvpr2023/code_release/PerfPredRecV2/"

conda activate swiftnet-pp-v2

python eval_attacks_n_noise.py \
--model_name SwiftNetRec \
--encoder resnet18 \
--rec_decoder swiftnet \
--model_state_name swiftnet_rn18 \
--weights_epoch 10 \
--dataset cityscapes \
--subset val \
--num_workers 2 \
--zeroMean 1 \
--epsilon 0 0.25 0.5 1 2 4 8 12 16 20 24 28 32 \