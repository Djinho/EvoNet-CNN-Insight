#!/bin/bash
#$ -cwd  # Set the working directory for the job to the current directory
#$ -pe smp 4  # Request 4 CPU cores
#$ -l h_rt=2:0:0  # Request 2 hours runtime (adjust as needed)
#$ -l h_vmem=4G  # Request 4GB RAM per core, total 16GB

# Load the Python module and ensure the correct environment
module load python/3.8

# Check TensorFlow installation
python -c "import tensorflow as tf; print('TensorFlow Version:', tf.__version__)"

# Run the experiment script
python run_experiment.py --dataset $DATASET --seed $SEED --output_dir $OUTPUT_DIR
