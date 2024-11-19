#!/bin/sh
#SBATCH -J FHNet-cub-resnet
#SBATCH -o /home/akun648/projects/MS_Freq_Net/result/out/train.out.%j
#SBATCH -e /home/akun648/projects/MS_Freq_Net/result/err/train.err.%j
#SBATCH --partition=gpu-a40
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24

source /home/akun648/anaconda3/bin/activate TransZero
python /home/akun648/projects/FHNet/experiments/CUB_fewshot_cropped/BiFRN/ResNet-12/train.py