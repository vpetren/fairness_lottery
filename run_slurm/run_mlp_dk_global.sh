#!/bin/bash
#SBATCH --ntasks=1 --cpus-per-task=1 --mem=15000M
#SBATCH -p gpu --gres=gpu:1
#SBATCH --time=4:00:00

python train_mlp.py --data data/trustpilot/denmark_demographics --hidden-dim 256 --embed-dim 128 --num-epochs 15 --batch-size 32 --pruning_pct 0.35 --model mlp --num-pruning-iter 20 --bidirectional --pruning_method magnitude --pruning_structure global
