
# EvoNet-CNN-Insight Project

## Overview
This project explores the efficacy of Convolutional Neural Networks (CNNs) in detecting selection across various evolutionary scenarios, ranging from temporally ancient to nearly neutral evolutionary contexts. The primary goal is to determine how effectively CNNs can identify selection pressures, particularly in challenging scenarios characterised by ancient and weak selection signals.

## Key Findings
- **Baseline/Low Complexity CNN**: Demonstrated the capability to detect selection in:
  - **Recent and Strong Selection**
  - **Ancient and Weak Selection**
  
- **Bayesian Optimized CNN**: Achieved improved accuracy across most scenarios, with accuracy rates ranging from **69.10% to 88.88%**. Notably, this optimization did not yield significant improvements in the **Recent and Weak** scenario.

## Project Structure
- **Autophagy_related_Genes**: Contains notebooks for testing the trained models (specifically those optimized with Bayesian optimization over 30 trials) against variants in autophagy-related genes. This folder includes testing scripts and plotting utilities.
- **Optimisation_notebooks**: [Brief description if needed]
- **Selection_Test**: Contains scripts for testing the trained models (30 trials of Bayesian optimization) on known variants under selection, including human pigmentation genes and lactose persistence genes. This folder also includes plotting utilities.
- **model_training**: This folder contains the main scripts and notebooks for training CNN models under different evolutionary scenarios.

  - **Ancient_moderate**
  - **Ancient_strong**
  - **Ancient_weak**
  - **intermediate_moderate**
  - **intermediate_strong**
  - **intermediate_weak**
  - **recent_moderate**
  - **recent_weak**
  - **recent_strong.ipynb**

  Each scenario contains the following key notebooks:
  - `Baseline.ipynb`: Use this notebook to train the CNN from scratch using baseline parameters.
  - `Optimisation.ipynb`: This notebook implements Bayesian optimization for hyperparameter tuning, run over 30 trials.
  - `Optimisation_2.ipynb`: A continuation of Bayesian optimization, extending the trials to 50 for more refined results.
  - `VGG16.ipynb`: Applies transfer learning using the VGG16 architecture.

## How to Use

### Running Simulations and Training Models
1. **Baseline Training:**
   - Navigate to the desired evolutionary scenario folder.
   - Open `Baseline.ipynb`.
   - Run all cells to train the CNN model using baseline settings.

2. **Bayesian Optimization:**
   - For hyperparameter optimization with 30 trials, open `Optimisation.ipynb`.
   - For extended optimization with 50 trials, open `Optimisation_2.ipynb`.
   - Run the cells in the notebook to optimize and train the model.

3. **Transfer Learning:**
   - If you want to apply transfer learning, navigate to `VGG16.ipynb` in the relevant evolutionary scenario folder.
   - Run all cells to fine-tune the CNN using the VGG16 architecture.

### Running Simulations
- If you do not have pre-generated simulation data and need to generate new simulations based on a specific evolutionary scenario:
  1. Open the desired notebook.
  2. Locate the code line containing:
     ```python
     # Run to make simulations
     import subprocess
     subprocess.call("bash ../../generate_dataset.sh (evolutionary_Scenario or params.txt file name)".split())
     ```
  3. Replace `(evolutionary_Scenario or params.txt file name)` with the appropriate parameters file name.
  4. Run the cell to generate the dataset.

### Testing and Plotting Results
- **Autophagy_related_Genes**: After training the models, you can test their performance on autophagy-related genes by using the notebooks in this folder. The scripts will plot the results to visualize the model's efficacy.
- **Selection_Test**: This folder contains notebooks for testing the model on known selected variants. The tests include genes related to human pigmentation and lactose persistence, and the results will be plotted for further analysis.

## Accessing and Using the Pre-trained Models

### Accessing the Models
The trained models, including those that underwent Bayesian optimization for 30 trials and the VGG16 trained models, are available via this [Google Drive link](https://drive.google.com/drive/folders/1sF8ERxF2d4opU3jBHMf1a3S6RAZ3k6WA?usp=sharing).

### Example Code for Loading and Using the Models
To use one of these models in your own environment, follow these steps:

1. **Download the Model**: First, download the desired model from the Google Drive link provided.
2. **Mount Google Drive**: If you are using Google Colab, mount Google Drive to access the model directly. For example:

    ```python
    from google.colab import drive
    drive.mount('/content/drive')
    ```

3. **Set the Model Path**: Modify the path to point to the downloaded model in your Drive or local system.

    ```python
    # Example code for using the model
    model_path = '/content/drive/My Drive/models/IM_Bayes_opt.keras'  # Adjust this path

    # Load the model
    model = tf.keras.models.load_model(model_path)

    # Make predictions and print results
    prediction = model.predict(gene_LCT.data, batch_size=None)[0][0]
    print(prediction)
    ```

4. **Adjust the Path**: If the model is stored locally, adjust the `model_path` accordingly:

    ```python
    model_path = '/path/to/downloaded/model/IM_Bayes_opt.keras'
    ```

5. **Run the Code**: Execute the code to load the model, make predictions, and interpret the results.

## Notes
- Ensure that you have all the required dependencies installed before running any notebooks.
- The evolutionary scenarios correspond to different levels of selection pressure and selection start times as outlined in the corresponding publications or project documentation.
- The report summarising the findings can be found in the CNN_Report.pdf 

## Contact
For further information or inquiries, please contact [Djinho] at [Djinho.itshary@outlook].
