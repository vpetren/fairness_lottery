#!/bin/bash
for i in {1..5}
do
   sbatch run_slurm/run_lstm_de_global.sh
   sbatch run_slurm/run_lstm_dk_global.sh
   sbatch run_slurm/run_lstm_uk_global.sh
   sbatch run_slurm/run_mlp_de_global.sh
   sbatch run_slurm/run_mlp_dk_global.sh
   sbatch run_slurm/run_mlp_uk_global.sh
done
