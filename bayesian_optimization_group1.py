import optuna
import tensorflow as tf
from tensorflow.keras import models, layers, regularizers, callbacks
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd
import logging
import skimage
import os
import joblib  # For checkpointing
import gzip




# Set random seeds for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

# Set up logging to record the script's progress and errors
logging.basicConfig(filename='error_group1.txt', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logging.info("Started bayesian_optimization_group1.py script")

# Execute ImaGene.py script to define ImaFile and other necessary functions/classes
try:
    exec(open('/data/home/ha231431/EvoNet-CNN-Insight/ImaGene.py').read())
    logging.info("Successfully executed ImaGene.py")
except Exception as e:
    logging.error(f"Error executing ImaGene.py: {e}")
    raise

# Ensure necessary variables are defined
if 'ImaFile' not in locals():
    logging.error("ImaFile is not defined. Check ImaGene.py script.")
    raise ImportError("ImaFile is not defined. Ensure ImaGene.py defines ImaFile.")

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
        mygene.convert(flip=True)
        mygene.subset(get_index_random(mygene))
        mygene.targets = to_binary(mygene.targets)
        data.append(mygene.data)
        targets.append(mygene.targets)
    data = np.concatenate(data, axis=0)
    targets = np.concatenate(targets, axis=0)
    return data, targets

def create_model(trial, input_shape):
    model = models.Sequential()
    num_layers = trial.suggest_int('num_layers', 1, 4)
    filters = [trial.suggest_int('num_filters_{}'.format(i), 16, 256) for i in range(num_layers)]
    kernel_size = trial.suggest_categorical('kernel_size', [(3, 3), (5, 5)])
    strides = trial.suggest_int('strides', 1, 3)
    l1_ratio = trial.suggest_float('l1_ratio', 1e-6, 1e-1, log=True)
    l2_ratio = trial.suggest_float('l2_ratio', 1e-6, 1e-1, log=True)
    dropout_rate = trial.suggest_float('dropout_rate', 0.0, 0.5)
    dense_units = trial.suggest_int('dense_units', 64, 512)
    activation = trial.suggest_categorical('activation', ['relu', 'tanh', 'leaky_relu'])
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-1, log=True)
    for i in range(num_layers):
        if i == 0:
            model.add(layers.Conv2D(filters=filters[i], kernel_size=kernel_size, strides=strides, activation=activation,
                                    kernel_regularizer=regularizers.l1_l2(l1=l1_ratio, l2=l2_ratio),
                                    padding='valid', input_shape=input_shape))
        else:
            model.add(layers.Conv2D(filters=filters[i], kernel_size=kernel_size, strides=strides, activation=activation,
                                    kernel_regularizer=regularizers.l1_l2(l1=l1_ratio, l2=l2_ratio),
                                    padding='valid'))
        model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(units=dense_units, activation=activation))
    model.add(layers.Dropout(rate=dropout_rate))
    model.add(layers.Dense(units=1, activation='sigmoid'))
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    return model

def objective(trial):
    try:
        model = create_model(trial, input_shape)
    except Exception as e:
        logging.error(f"Model creation failed for trial {trial.number}: {e}")
        raise optuna.exceptions.TrialPruned()
    batch_size = 64
    epochs = 10
    early_stopping = callbacks.EarlyStopping(monitor='val_loss', patience=3)
    try:
        history = model.fit(X_train, y_train, validation_data=(X_val, y_val),
                            batch_size=batch_size, epochs=epochs,
                            callbacks=[early_stopping], verbose=2)
    except Exception as e:
        logging.error(f"Model training failed for trial {trial.number}: {e}")
        raise optuna.exceptions.TrialPruned()
    val_acc = history.history['val_accuracy'][-1]
    return val_acc

# Define the datasets and their paths
datasets = {
    "AM": "/data/home/ha231431/EvoNet-CNN-Insight/AM",
    "AS": "/data/home/ha231431/EvoNet-CNN-Insight/AS",
    "AW": "/data/home/ha231431/EvoNet-CNN-Insight/AW"
}

optimal_params = {}
batches = 10  # Use 10 batches for optimization

for scenario, path in datasets.items():
    logging.info(f"Starting optimization for scenario: {scenario}")

    # Load and preprocess data
    try:
        data, targets = load_and_preprocess(path, batches)
    except Exception as e:
        logging.error(f"Error loading data for {scenario}\n{e}")
        continue

    input_shape = data.shape[1:]

    # Split data
    X_train, X_val, y_train, y_val = train_test_split(data, targets, test_size=0.1, random_state=RANDOM_SEED)

    # Optimize the model using Optuna
    try:
        study_path = f"study_{scenario}.pkl"
        if os.path.exists(study_path):
            study = joblib.load(study_path)
        else:
            study = optuna.create_study(direction='maximize')

        study.optimize(objective, n_trials=25)

        # Save the study after each trial to avoid losing progress
        joblib.dump(study, study_path)

        optimal_params[scenario] = study.best_trial.params

        logging.info(f"Optimal parameters for scenario {scenario}: {optimal_params[scenario]}")
    except Exception as e:
        logging.error(f"Error during optimization for {scenario}\n{e}")

# Save the optimal parameters for each scenario to a CSV file
optimal_params_df = pd.DataFrame.from_dict(optimal_params, orient='index')
optimal_params_df.to_csv("optimal_params_group1.csv")

logging.info("Completed bayesian_optimization_group1.py script")
