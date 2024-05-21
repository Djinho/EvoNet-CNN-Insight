#!/bin/bash

datasets=("Early_Weak" "Early_Moderate" "Early_Strong" "Mid_Weak" "Mid_Moderate" "Mid_Strong" "Late_Weak" "Late_Moderate" "Late_Strong")

# Read seeds from the file generated by generate_seeds.py
mapfile -t seeds < seeds.txt

# Limit the number of runs per scenario to 3 (Adjustable to 5 if resources allow)
num_runs=3

for seed_idx in $(seq 0 $((num_runs-1))); do
  seed=${seeds[seed_idx]}
  
  for dataset in "${datasets[@]}"; do
    qsub job_script.sh "Datasets/${dataset}" ${seed} "./Baseline_Experiments/Results"
  done
done
