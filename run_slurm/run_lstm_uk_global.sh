#!/bin/bash
#SBATCH --ntasks=1 --cpus-per-task=1 --mem=15000M
#SBATCH -p gpu --gres=gpu:titanrtx:1
#SBATCH --time=8:00:00

python train_mlp.py --data data/trustpilot/united_kingdom_demographics --hidden-dim 256 --embed-dim 128 --num-epochs 10 --batch-size 32 --pruning_pct 0.35 --model lstm --num-pruning-iter 20 --bidirectional --pruning_method magnitude --pruning_structure global
