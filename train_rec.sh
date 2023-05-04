#!/bin/bash -l

#SBATCH --job-name=trn_SNR
#SBATCH --partition=gpu         # gpu or debug
#SBATCH --time=1:20:00        # 100:20:00 (gpu) or 00:20:00 (debug)
#SBATCH --begin=now
#SBATCH --gres=gpu:1080:1
#SBATCH --mem=20000M		    # was 50000/20000/15000
#SBATCH --cpus-per-task=4       # 2 or 4

# Replace with your dataset or checkpoint path
export IFN_DIR_DATASET=/beegfs/data/shared
export IFN_DIR_CHECKPOINT="${PWD}/../../experiments/"

conda activate swiftnet-pp-v2

python train_swiftnet_rec.py \
--model_name SwiftNetRec \
--encoder resnet18 \
--savedir swiftnet-rn18 \
--dataset cityscapes \
--zeromean 1 \
--batch_size_train 8 \
--num_epochs 10 \
--rec_decoder swiftnet \
--lateral 1 \
--load_model_state_name ../SwiftNet/swiftnet_baseline/