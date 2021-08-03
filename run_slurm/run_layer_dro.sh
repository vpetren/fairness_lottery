#!/bin/bash
for i in {1..5}
do
   sbatch run_slurm/run_mlp_de_layer_dro.sh
   sbatch run_slurm/run_mlp_dk_layer_dro.sh
   sbatch run_slurm/run_mlp_uk_layer_dro.sh
   sbatch run_slurm/run_mlp_cc_layer_dro.sh
done