# EvoNet-CNN-Insight
Dataset Generation

This section guides you through generating the datasets required for training models using the EvoNet-CNN-Insight framework. The datasets simulate various evolutionary scenarios and selection strengths.
Prerequisites

Ensure Java is installed on your system to run msms.jar.
Directory Structure

The repository should resemble the following structure:

EvoNet-CNN-Insight/
│
├── msms.jar
├── generate_dataset.sh
├── Dataset_Generation/
│   ├── Gen_Dataset.ipynb
│   ├── params_early_weak.txt
│   ├── params_early_moderate.txt
│   ├── params_early_strong.txt
│   ├── params_mid_weak.txt
│   ├── params_mid_moderate.txt
│   ├── params_mid_strong.txt
│   ├── params_late_weak.txt
│   ├── params_late_moderate.txt
│   ├── params_late_strong.txt
└── Datasets/ (generated datasets will be stored here)

Generating the Datasets

    Clone the Repository:

    Clone the EvoNet-CNN-Insight repository into your local environment:

!git clone https://github.com/Djinho/EvoNet-CNN-Insight.git

Navigate to the Dataset_Generation Directory:

Change the working directory to Dataset_Generation:

%cd /content/EvoNet-CNN-Insight/Dataset_Generation

Generate the Datasets:

Use the Gen_Dataset.ipynb notebook to execute the dataset generation scripts. This notebook automates the dataset generation process using generate_dataset.sh and the parameter files.

Open the Gen_Dataset.ipynb notebook and run the cells to generate all datasets. The notebook executes the following main script:

    import subprocess

    # List of parameter files
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

    # Execute the generate_dataset.sh script with each parameter file
    for params_file in params_files:
        print(f"Running simulation for {params_file}")
        subprocess.call(["bash", "../generate_dataset.sh", params_file])

    print("All simulations completed.")

Parameter Files Overview

Parameter files are provided to simulate various evolutionary scenarios and selection strengths. Each parameter file customizes the demographic model, locus size, selection coefficients, and timing.

    Early Selection: Simulates weak, moderate, and strong selection pressures occurring approximately 10,000 years ago.
        params_early_weak.txt
        params_early_moderate.txt
        params_early_strong.txt

    Mid Selection: Simulates weak, moderate, and strong selection pressures occurring approximately 50,000 years ago.
        params_mid_weak.txt
        params_mid_moderate.txt
        params_mid_strong.txt

    Late Selection: Simulates weak, moderate, and strong selection pressures occurring approximately 100,000 years ago.
        params_late_weak.txt
        params_late_moderate.txt
        params_late_strong.txt

Each parameter file sets the simulation configurations, storing the generated data in the Datasets directory under folders named by the scenario (e.g., Datasets/Early_Weak).

By following this guide, you can successfully generate all required datasets for your models. If you have any questions or need further assistance, please refer to the documentation or contact the repository maintainers.

