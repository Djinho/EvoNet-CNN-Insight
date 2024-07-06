import optuna
import tensorflow as tf
from tensorflow.keras import models, layers, regularizers, callbacks
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import numpy as np
import pandas as pd
import logging
import skimage

# Set random seeds for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

# Set up logging
logging.basicConfig(filename='error.txt', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logging.info("Started bayesian_optimization.py script")

def load_and_preprocess(simulations_folder, batches):
    """
    Load and preprocess simulation data from multiple batches.

    Args:
        simulations_folder (str): Path to the folder containing simulation data.
        batches (int): Number of batches to process.

    Returns:
        tuple: Concatenated data and targets arrays.
    """
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

def build_model(trial, input_shape):
    """
    Build a CNN model based on the hyperparameters suggested by the Optuna trial.

    Args:
        trial (optuna.trial.Trial): Optuna trial object.
        input_shape (tuple): Shape of the input data.

    Returns:
        tensorflow.keras.Model: Compiled Keras model.
    """
    num_layers = trial.suggest_int('num_layers', 1, 4)
    filters = [trial.suggest_int(f'num_filters_{i}', 16, 256) for i in range(num_layers)]
    kernel_size = trial.suggest_categorical('kernel_size', [(3, 3), (5, 5), (7, 7)])
    strides = trial.suggest_int('strides', 1, 3)
    l1_ratio = trial.suggest_float('l1_ratio', 1e-6, 1e-1, log=True)
    l2_ratio = trial.suggest_float('l2_ratio', 1e-6, 1e-1, log=True)
    dropout_rate = trial.suggest_float('dropout_rate', 0.0, 0.5)
    dense_units = trial.suggest_int('dense_units', 64, 512)
    activation = trial.suggest_categorical('activation', ['relu', 'tanh', 'leaky_relu'])
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-1, log=True)

    model = models.Sequential()
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

def is_valid_configuration(trial, input_shape):
    """
    Check if the model configuration is valid to prevent negative dimensions.

    Args:
        trial (optuna.trial.Trial): Optuna trial object.
        input_shape (tuple): Shape of the input data.

    Returns:
        bool: True if the configuration is valid, False otherwise.
    """
    num_layers = trial.suggest_int('num_layers', 1, 4)
    kernel_size = trial.suggest_categorical('kernel_size', [(3, 3), (5, 5), (7, 7)])
    strides = trial.suggest_int('strides', 1, 3)

    height, width = input_shape[0], input_shape[1]
    for _ in range(num_layers):
        height = (height - kernel_size[0]) // strides + 1
        width = (width - kernel_size[1]) // strides + 1
        if height <= 0 or width <= 0:
            return False
        height //= 2
        width //= 2
    return True

def objective(trial, data, targets):
    """
    Objective function for Bayesian Optimization with Optuna.

    Args:
        trial (optuna.trial.Trial): Optuna trial object.
        data (np.array): Input data.
        targets (np.array): Target labels.

    Returns:
        float: Validation accuracy of the model.
    """
    if not is_valid_configuration(trial, data.shape[1:]):
        logging.error(f"Invalid model configuration for trial params: {trial.params}")
        raise optuna.exceptions.TrialPruned()

    try:
        logging.info(f"Creating model with params: {trial.params}")
        model = build_model(trial, data.shape[1:])
    except ValueError as e:
        logging.error(f"Error creating model with trial params: {trial.params}\n{e}")
        raise optuna.exceptions.TrialPruned()

    x_train, x_val, y_train, y_val = train_test_split(data, targets, test_size=0.1, random_state=RANDOM_SEED)

    early_stopping = callbacks.EarlyStopping(monitor='val_loss', patience=3)
    lr_scheduler = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, min_lr=1e-7)

    logging.info("Training the model")
    model.fit(x_train, y_train, epochs=10, batch_size=64, validation_data=(x_val, y_val),
              callbacks=[early_stopping, lr_scheduler], verbose=2)

    logging.info("Evaluating the model")
    y_pred = (model.predict(x_val) > 0.5).astype("int32")

    accuracy = accuracy_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred)
    precision = precision_score(y_val, y_pred)
    recall = recall_score(y_val, y_pred)

    logging.info(f"Model metrics - Accuracy: {accuracy}, F1-score: {f1}, Precision: {precision}, Recall: {recall}")

    # Return the accuracy as the optimization metric
    return accuracy

# Define the datasets and their paths
datasets = {
    "AM": "/data/home/ha231431/EvoNet-CNN-Insight/AM",
    "AS": "/data/home/ha231431/EvoNet-CNN-Insight/AS",
    "AW": "/data/home/ha231431/EvoNet-CNN-Insight/AW",
    "IM": "/data/home/ha231431/EvoNet-CNN-Insight/IM",
    "IS": "/data/home/ha231431/EvoNet-CNN-Insight/IS",
    "IW": "/data/home/ha231431/EvoNet-CNN-Insight/IW",
    "RM": "/data/home/ha231431/EvoNet-CNN-Insight/RM",
    "RS": "/data/home/ha231431/EvoNet-CNN-Insight/RS",
    "RW": "/data/home/ha231431/EvoNet-CNN-Insight/RW"
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

    # Optimize the model using Optuna
    try:
        study = optuna.create_study(direction='maximize')
        study.optimize(lambda trial: objective(trial, data, targets), n_trials=50)

        optimal_params[scenario] = study.best_trial.params

        logging.info(f"Optimal parameters for scenario {scenario}: {optimal_params[scenario]}")
    except Exception as e:
        logging.error(f"Error during optimization for {scenario}\n{e}")

# Save the optimal parameters for each scenario to a CSV file
optimal_params_df = pd.DataFrame.from_dict(optimal_params, orient='index')
optimal_params_df.to_csv("optimal_params.csv")

logging.info("Completed bayesian_optimization.py script")

