import optuna
import tensorflow as tf
from tensorflow.keras import models, layers, regularizers
from sklearn.model_selection import train_test_split
import numpy as np
import _pickle as pickle
import gzip
import os
import sys
import skimage.transform
import logging

# Set random seeds for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

# Set up logging
logging.basicConfig(filename='error.txt', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Log script start
logging.info("Started bayesian_optimization.py script")

# Import ImaGene
exec(open('/data/home/ha231431/EvoNet-CNN-Insight/Model_training_3/ImaGene.py').read())

# Function to preprocess and load simulation data from multiple batches
def load_and_preprocess(simulations_folder, batches):
    data, targets = [], []
    for i in range(1, batches + 1):
        sim_folder = f'{simulations_folder}/Simulations{i}'
        logging.info(f"Processing: {sim_folder}")
        myfile = ImaFile(simulations_folder=sim_folder, nr_samples=198, model_name='Marth-3epoch-CEU')
        mygene = myfile.read_simulations(parameter_name='selection_coeff_hetero', max_nrepl=2000)
        mygene.majorminor()
        mygene.filter_freq(0.01)
        mygene.sort('rows_freq')
        mygene.resize((198, 192))
        mygene.convert()
        mygene.classes = np.array([0, 1])
        mygene.subset(get_index_classes(mygene.targets, mygene.classes))
        mygene.subset(get_index_random(mygene))
        mygene.targets = to_binary(mygene.targets)
        data.append(mygene.data)
        targets.append(mygene.targets)

    data = np.concatenate(data, axis=0)
    targets = np.concatenate(targets, axis=0)
    logging.info(f"Total data shape: {data.shape}")
    logging.info(f"Total targets shape: {targets.shape}")
    return data, targets

# Define the objective function for Bayesian Optimization
def objective(trial, data, targets):
    num_filters_1 = trial.suggest_int('num_filters_1', 16, 64)
    num_filters_2 = trial.suggest_int('num_filters_2', 32, 128)
    num_filters_3 = trial.suggest_int('num_filters_3', 32, 128)
    kernel_size = trial.suggest_categorical('kernel_size', [(3, 3), (5, 5)])
    strides = trial.suggest_int('strides', 1, 2)
    l1_ratio = trial.suggest_float('l1_ratio', 1e-6, 1e-2, log=True)
    l2_ratio = trial.suggest_float('l2_ratio', 1e-6, 1e-2, log=True)
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)

    try:
        logging.info(f"Creating model with params: {trial.params}")
        model = models.Sequential([
            layers.Conv2D(filters=num_filters_1, kernel_size=kernel_size, strides=strides, activation='relu',
                          kernel_regularizer=regularizers.l1_l2(l1=l1_ratio, l2=l2_ratio),
                          input_shape=data.shape[1:]),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Conv2D(filters=num_filters_2, kernel_size=kernel_size, strides=strides, activation='relu',
                          kernel_regularizer=regularizers.l1_l2(l1=l1_ratio, l2=l2_ratio)),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Conv2D(filters=num_filters_3, kernel_size=kernel_size, strides=strides, activation='relu',
                          kernel_regularizer=regularizers.l1_l2(l1=l1_ratio, l2=l2_ratio)),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Flatten(),
            layers.Dense(128, activation='relu'),
            layers.Dense(1, activation='sigmoid')
        ])
    except ValueError as e:
        logging.error(f"Error creating model with trial params: {trial.params}\n{e}")
        raise optuna.exceptions.TrialPruned()

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

    x_train, x_val, y_train, y_val = train_test_split(data, targets, test_size=0.1, random_state=RANDOM_SEED)

    logging.info("Training the model")
    model.fit(x_train, y_train, epochs=3, batch_size=64, validation_data=(x_val, y_val), verbose=0)

    logging.info("Evaluating the model")
    score = model.evaluate(x_val, y_val, verbose=0)
    accuracy = score[1]

    logging.info(f"Model accuracy: {accuracy}")

    return accuracy

# Define the datasets and their paths
datasets = {
    "AM": "/data/home/ha231431/EvoNet-CNN-Insight/AM",
    "AS": "/data/home/ha231431/EvoNet-CNN-Insight/AS",
    "IM": "/data/home/ha231431/EvoNet-CNN-Insight/IM",
    "IS": "/data/home/ha231431/EvoNet-CNN-Insight/IS",
    "RM": "/data/home/ha231431/EvoNet-CNN-Insight/RW"
}

optimal_params = {}
batches = 5  # Use a subset of batches for optimization (you can adjust this number)

for scenario, path in datasets.items():
    logging.info(f"Starting optimization for scenario: {scenario}")

    # Load and preprocess data
    try:
        data, targets = load_and_preprocess(path, batches)
    except Exception as e:
        logging.error(f"Error loading data for {scenario}\n{e}")
        continue

    # Optimize the model using optuna
    try:
        study = optuna.create_study(direction='maximize')
        study.optimize(lambda trial: objective(trial, data, targets), n_trials=50)

        optimal_params[scenario] = study.best_trial.params

        logging.info(f"Optimal parameters for scenario {scenario}: {optimal_params[scenario]}")
    except Exception as e:
        logging.error(f"Error during optimization for {scenario}\n{e}")

# Save the optimal parameters for each scenario
with open("optimal_params.pkl", "wb") as f:
    pickle.dump(optimal_params, f)

logging.info("Completed bayesian_optimization.py script")
