
#!/bin/bash
#$ -cwd           # Set the working directory for the job to the current directory
#$ -pe smp 1      # Request 1 core
#$ -l h_rt=1:0:0  # Request 1 hour runtime
#$ -l h_vmem=1G   # Request 1GB RAM

# Load any necessary modules (if needed)
# module load example_module

# Run the dataset generation commands
bash generate_datasets.sh params_Ant_moderate.txt
bash generate_datasets.sh params_Ant_weak.txt
bash generate_datasets.sh params_Ant_strong.txt
bash generate_datasets.sh params_INT_moderat.txt
bash generate_datasets.sh params_INT_strong.txt
bash generate_datasets.sh params_INT_weak.txt
bash generate_datasets.sh params_REC_moderate.txt
bash generate_datasets.sh params_REC_strong.txt
bash generate_datasets.sh params_REC_Weak.txt

