#!/bin/bash
for i in {1..5}
do
   sbatch run_slurm/run_mlp_de_layer.sh
   sbatch run_slurm/run_mlp_dk_layer.sh
   sbatch run_slurm/run_mlp_uk_layer.sh
   sbatch run_slurm/run_mlp_cc_layer.sh
done