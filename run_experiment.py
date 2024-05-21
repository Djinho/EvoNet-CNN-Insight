import os
import json
import argparse
import numpy as np
import tensorflow as tf
from keras import models, layers, regularizers
from keras.callbacks import Callback
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve
import time
from imagene import ImaFile, load_imagene, to_binary, get_index_random, ImaNet, get_index_classes  # Ensured correct import

class MetricsLogger(Callback):
    def __init__(self):
        super().__init__()
        self.logs = {"loss":[], "val_loss":[]}

    def on_epoch_end(self, epoch, logs=None):
        self.logs["loss"].append(logs.get("loss"))
        self.logs["val_loss"].append(logs.get("val_loss"))

def load_simulation_data(simulation_path, nr_samples):
    file_sim = ImaFile(simulations_folder=simulation_path, nr_samples=nr_samples, model_name='Marth-3epoch-CEU')
    gene_sim = file_sim.read_simulations(parameter_name='selection_coeff_hetero', max_nrepl=2000)
    gene_sim.filter_freq(0.01)
    gene_sim.sort('rows_freq')
    gene_sim.resize((198, 192))
    gene_sim.convert(flip=True)
    gene_sim.targets = to_binary(gene_sim.targets)
    gene_sim.subset(get_index_random(gene_sim))  # Randomize the order of samples
    return gene_sim

def build_cnn_model(input_shape):
    model = models.Sequential([
        layers.Conv2D(filters=32, kernel_size=(3,3), strides=(1,1), activation='relu', 
                      kernel_regularizer=regularizers.l1_l2(l1=0.005, l2=0.005), padding='valid', 
                      input_shape=input_shape),
        layers.MaxPooling2D(pool_size=(2,2)),
        layers.Conv2D(filters=32, kernel_size=(3,3), strides=(1,1), activation='relu', 
                      kernel_regularizer=regularizers.l1_l2(l1=0.005, l2=0.005), padding='valid'),
        layers.MaxPooling2D(pool_size=(2,2)),
        layers.Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), activation='relu', 
                      kernel_regularizer=regularizers.l1_l2(l1=0.005, l2=0.005), padding='valid'),
        layers.MaxPooling2D(pool_size=(2,2)),
        layers.Flatten(),
        layers.Dense(units=128, activation='relu'),
        layers.Dense(units=1, activation='sigmoid')
    ])
    model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def train_model(model, data, targets, epochs=1, batch_size=64, validation_split=0.10, log_metrics=None):
    return model.fit(data, targets, batch_size=batch_size, epochs=epochs, verbose=1, validation_split=validation_split, callbacks=[log_metrics])

def evaluate_model(model, test_data, test_targets):
    return model.evaluate(test_data, test_targets, batch_size=None, verbose=0)

def inference_time_model(model, data):
    start_time = time.time()
    model.predict(data)
    end_time = time.time()
    return end_time - start_time

def model_size(model):
    return model.count_params()

def run_experiment(simulation_folders, seed, output_dir='./Baseline_Experiments/Results'):
    np.random.seed(seed)
    tf.random.set_seed(seed)
    
    model = None
    net_LCT = ImaNet(name='[C32+P]x2+[C64+P]+D128')  # Initialize ImaNet
    log_metrics = MetricsLogger()
    
    # Train on all simulation batches except the last one
    for i, sim_folder in enumerate(simulation_folders[:-1]):  # Exclude the last one
        print(f'Starting Simulation {i + 1}')
        gene_sim = load_simulation_data(sim_folder, nr_samples=198)
        
        if i == 0:
            model = build_cnn_model(gene_sim.data.shape[1:])
        
        start_time = time.time()
        score = train_model(model, gene_sim.data, gene_sim.targets, epochs=1, batch_size=64, validation_split=0.10, log_metrics=log_metrics)
        training_time = time.time() - start_time
        net_LCT.update_scores(score)
        
        print(f'Completed Training on Simulation {i + 1}')
    
    # Perform final evaluation on the last batch
    test_sim_folder = simulation_folders[-1]  # Last simulation folder
    gene_sim_test = load_simulation_data(test_sim_folder, nr_samples=198)
    test_scores = evaluate_model(model, gene_sim_test.data, gene_sim_test.targets)

    # Calculate additional metrics
    y_pred_prob = model.predict(gene_sim_test.data)
    y_pred = (y_pred_prob > 0.5).astype("int32")
    accuracy = accuracy_score(gene_sim_test.targets.argmax(axis=-1), y_pred)
    precision = precision_score(gene_sim_test.targets.argmax(axis=-1), y_pred)
    recall = recall_score(gene_sim_test.targets.argmax(axis=-1), y_pred)
    f1 = f1_score(gene_sim_test.targets.argmax(axis=-1), y_pred)
    auc_roc = roc_auc_score(gene_sim_test.targets, y_pred_prob)
    fpr, tpr, thresholds = roc_curve(gene_sim_test.targets, y_pred_prob)
    
    net_LCT.test = test_scores
    
    # Save the ROC curve data
    roc_curve_data = {
        'fpr': fpr.tolist(),
        'tpr': tpr.tolist(),
        'thresholds': thresholds.tolist()
    }
    
    # Save the model and network
    model_path = os.path.join(output_dir, f'model_seed_{seed}.h5')
    save_model(model, model_path)
    
    net_LCT_path = os.path.join(output_dir, f'net_LCT_seed_{seed}.binary')
    net_LCT.save(net_LCT_path)
    
    # Save results for further analysis
    results_path = os.path.join(output_dir, f'results_seed_{seed}.json')
    results = {
        'seed': seed,
        'train_scores': net_LCT.train_scores,
        'loss_decay': log_metrics.logs,
        'test_accuracy': accuracy,
        'test_precision': precision,
        'test_recall': recall,
        'test_f1_score': f1,
        'test_auc_roc': auc_roc,
        'inference_time': inference_time,
        'model_size': total_model_size,
        'training_time': training_time,
        'roc_curve_data': roc_curve_data  # Save ROC curve data
    }
    with open(results_path, 'w') as f:
        json.dump(results, f)
    
    print(f'Completed Experiment for Seed {seed}')

def parse_arguments():
    parser = argparse.ArgumentParser(description="Run experiments on datasets with different seeds")
    parser.add_argument('--dataset', type=str, required=True, help="Path to the dataset folder")
    parser.add_argument('--seed', type=int, required=True, help="Seed for reproducibility")
    parser.add_argument('--output_dir', type=str, default='./Baseline_Experiments/Results', help="Directory to output the results")
    return parser.parse_args()

def main():
    args = parse_arguments()
    datasets = [os.path.join(args.dataset, f'Simulations{str(i)}') for i in range(1, 11)]
    run_experiment(datasets, args.seed, args.output_dir)

if __name__ == "__main__":
    main()
