
import optuna
import tensorflow as tf
from tensorflow.keras import models, layers, regularizers, callbacks
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd
import logging
import skimage
import os
import gzip
import time
import pathlib

# Path to ImaGene.py
IMAGENE_PATH = '/data/home/ha231431/EvoNet-CNN-Insight/ImaGene.py'

# Configure logging to file
logging.basicConfig(filename='Recent.log', level=logging.INFO,
                    format='%(asctime)s:%(levelname)s:%(message)s')

# Execute ImaGene.py script to define ImaFile and other necessary functions/classes
try:
    exec(open(IMAGENE_PATH).read())
    logging.info("Successfully executed ImaGene.py")
except Exception as e:
    logging.error(f"Error executing ImaGene.py: {e}")
    raise

# Ensure necessary variables are defined
if 'ImaFile' not in locals():
    logging.error("ImaFile is not defined. Check ImaGene.py script.")
    raise ImportError("ImaFile is not defined. Ensure ImaGene.py defines ImaFile.")

# Define the dataset paths
SIMULATIONS_FOLDERS = [
    '/data/home/ha231431/EvoNet-CNN-Insight/RW',
    '/data/home/ha231431/EvoNet-CNN-Insight/RM',
    '/data/home/ha231431/EvoNet-CNN-Insight/RS'
]

def preprocess_data(simulations_folder, batch_number):
    file_sim = ImaFile(simulations_folder=os.path.join(simulations_folder, f'Simulations{batch_number}'), nr_samples=198, model_name='Marth-3epoch-CEU')
    gene_sim = file_sim.read_simulations(parameter_name='selection_coeff_hetero', max_nrepl=2000)
    gene_sim.filter_freq(0.01)
    gene_sim.sort('rows_freq')
    gene_sim.resize((198, 192))
    gene_sim.convert(flip=True)
    gene_sim.subset(get_index_random(gene_sim))
    gene_sim.targets = to_binary(gene_sim.targets)
    return gene_sim

def objective(trial, simulations_folder, epochs=1):
    # Set random seed
    np.random.seed(42)
    tf.random.set_seed(42)

    filters_1 = trial.suggest_int('filters_1', 32, 128)
    filters_2 = trial.suggest_int('filters_2', 32, 128)
    filters_3 = trial.suggest_int('filters_3', 64, 256)
    kernel_size = trial.suggest_int('kernel_size', 3, 7)
    l1 = trial.suggest_loguniform('l1', 1e-6, 1e-2)
    l2 = trial.suggest_loguniform('l2', 1e-6, 1e-2)
    dense_units = trial.suggest_int('dense_units', 64, 512)
    dropout_rate = trial.suggest_uniform('dropout_rate', 0.1, 0.5)
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-3)

    model = None
    for batch_number in range(1, 10):  # Use 9 batches for training
        gene_sim = preprocess_data(simulations_folder, batch_number)

        if batch_number == 1:
            model = models.Sequential([
                layers.Conv2D(filters=filters_1, kernel_size=(kernel_size, kernel_size), strides=(1, 1), activation='relu', kernel_regularizer=regularizers.l1_l2(l1=l1, l2=l2), padding='valid', input_shape=gene_sim.data.shape[1:]),
                layers.MaxPooling2D(pool_size=(2, 2)),
                layers.Conv2D(filters=filters_2, kernel_size=(kernel_size, kernel_size), strides=(1, 1), activation='relu', kernel_regularizer=regularizers.l1_l2(l1=l1, l2=l2), padding='valid'),
                layers.MaxPooling2D(pool_size=(2, 2)),
                layers.Conv2D(filters=filters_3, kernel_size=(kernel_size, kernel_size), strides=(1, 1), activation='relu', kernel_regularizer=regularizers.l1_l2(l1=l1, l2=l2), padding='valid'),
                layers.MaxPooling2D(pool_size=(2, 2)),
                layers.Flatten(),
                layers.Dense(units=dense_units, activation='relu'),
                layers.Dropout(rate=dropout_rate),
                layers.Dense(units=1, activation='sigmoid')
            ])

            optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
            model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

        early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
        model.fit(gene_sim.data, gene_sim.targets, batch_size=64, epochs=epochs, verbose=1, validation_split=0.1, callbacks=[early_stopping])

    # Evaluate the model on the 10th batch
    gene_sim = preprocess_data(simulations_folder, 10)
    evaluation = model.evaluate(gene_sim.data, gene_sim.targets, verbose=1)
    return evaluation[1]

def run_optimization():
    start_time = time.time()

    for simulations_folder in SIMULATIONS_FOLDERS:
        logging.info(f"Starting optimization for {simulations_folder}")
        print(f"Starting optimization for {simulations_folder}")

        study = optuna.create_study(direction='maximize')
        study.optimize(lambda trial: objective(trial, simulations_folder), n_trials=50)  # Set to 50 trials

        try:
            trial = study.best_trial
            print('Best trial:')
            print('Value: {}'.format(trial.value))
            print('Params: ')
            for key, value in trial.params.items():
                print('    {}: {}'.format(key, value))

            logging.info('Best trial:')
            logging.info('Value: {}'.format(trial.value))
            logging.info('Params: ')
            for key, value in trial.params.items():
                logging.info('    {}: {}'.format(key, value))

        except ValueError as e:
            logging.error(f"No completed trials: {e}")

    elapsed_time = time.time() - start_time
    print(f"Elapsed time: {elapsed_time:.2f} seconds")
    logging.info(f"Elapsed time: {elapsed_time:.2f} seconds")

if __name__ == '__main__':
    run_optimization()
