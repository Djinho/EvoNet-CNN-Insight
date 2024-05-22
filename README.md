# EvoNet-CNN-Insight

Generating Data
Overview
The generation of synthetic datasets is pivotal for evaluating and validating machine learning models, particularly Convolutional Neural Networks (CNNs) applied to bioinformatics. This section describes the high-level methodology and parameters used to create realistic genomic datasets, reflecting diverse evolutionary scenarios.

Simulation Environment
All simulations are conducted using the msms toolkit, a robust coalescent simulation software that accommodates both neutral and selective evolutionary pressures. The simulations were executed within a Google Colab environment, leveraging its computational resources for efficient data generation.

Configuration and Parameters
The following configuration parameters ensure a reproducible and controlled simulation environment:

DIRMSMS: The path to the msms.jar executable.
DIRDATA: The directory where the generated datasets are stored.
NREF: Reference population size, set to 10,000 individuals.
DEMO: Demographic model specifying population size changes over time, utilizing the 3-epoch model by Marth et al. 2004.
LEN: Length of each locus, set to 80,000 base pairs.
THETA: Population mutation rate.
RHO: Population recombination rate.
NCHROMS: Number of chromosomes sampled, set to 198.
SELPOS: Position of selection on the locus, at 50% of the locus length.
FREQ: Initial allele frequency at the time of selection, set to 1%.
SELRANGE: Range of selection coefficients simulated, producing different strengths of selection.
TIMERANGE: Different timings of selection onset, representing various evolutionary scenarios.
NREPL: Number of replicates for each scenario, typically set to 1000.
NBATCH: Number of batches to split the simulations into, enhancing manageability.
NTHREADS: Number of threads utilized for parallel processing, maximizing computational efficiency.
Execution Steps
To generate the datasets, follow these steps:

Clone the Repository:

git clone https://github.com/Djinho/EvoNet-CNN-Insight.git
Navigate to the Dataset Generation Directory:

cd EvoNet-CNN-Insight/Dataset_Generation
Import Required Libraries:

import subprocess
import numpy as np
Set a Base Seed for Reproducibility:

base_seed = 42
np.random.seed(base_seed)
List of Parameter Files and Scenarios:

params_files = [
    "params_early_weak.txt",
    "params_early_moderate.txt",
    "params_early_strong.txt",
    "params_mid_weak.txt",
    "params_mid_moderate.txt",
    "params_mid_strong.txt",
    "params_late_weak.txt",
    "params_late_moderate.txt",
    "params_late_strong.txt",
]
Execute the generate_dataset.sh Script:

for params_file in params_files:
    # Create a reproducible seed
    seed = np.random.randint(0, 10000)

    print(f"Running simulation for {params_file} with seed {seed}")
    subprocess.call(["bash", "../generate_dataset.sh", params_file, str(seed)])

print("All simulations completed.")
This script ensures that simulation datasets are generated for each specified parameter file, with reproducibility guaranteed through a base seed.

Conclusion
This rigorous approach to data generation, integrating advanced simulation tools with a controlled yet flexible parameter setup, ensures that the synthetic datasets produced are both realistic and diverse. These datasets are foundational for training and validating CNN models, advancing the field of evolutionary genomics through robust computational methods.
 
