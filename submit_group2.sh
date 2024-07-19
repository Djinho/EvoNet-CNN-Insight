#!/bin/bash
#$ -cwd                 # Set the working directory for the job to the current directory
#$ -j y                 # Join stdout and stderr
#$ -pe smp 4            # Request 4 CPU cores
#$ -l h_rt=240:0:0      # Request 168 hours runtime (7 days)
#$ -l h_vmem=16G        # Request 16GB RAM / core, i.e., 64GB total
#$ -m e                 # Send an email once the job is completed
#$ -M ha231431@qmul.ac.uk  # Specify your email address for notifications

# Load necessary modules
module load python/3.8.5

# Activate your Python virtual environment
source /data/home/ha231431/EvoNet-CNN-Insight/venv/bin/activate

# Ensure TensorFlow and scikit-image are installed
pip install --upgrade tensorflow
pip install scikit-image

# Run the Python script for Group 2
python bayesian_optimization_group2.py
