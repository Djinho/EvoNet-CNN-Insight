import json
import glob
import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

def mean_confidence_interval(data, confidence=0.95):
    n = len(data)
    m = np.mean(data)
    se = stats.sem(data)
    h = se * stats.t.ppf((1 + confidence) / 2., n-1)
    return m, m-h, m+h

# Load results
results_files = glob.glob('./Baseline_Experiments/Results/*.json')
all_metrics = []

for file in results_files:
    with open(file) as f:
        data = json.load(f)
        all_metrics.append(data)

df = pd.DataFrame(all_metrics)

# Compute mean and confidence intervals
metrics_summary = df.groupby('dataset').agg(['mean', 'std'])
metrics_with_ci = {}

for metric in ['test_accuracy', 'test_precision', 'test_recall', 'test_f1_score', 'test_auc_roc', 'inference_time', 'model_size', 'training_time']:
    metric_mean_ci = df.groupby('dataset')[metric].apply(lambda x: mean_confidence_interval(x))
    metrics_with_ci[metric] = metric_mean_ci

# Compute mean and standard deviation for ROC curves
roc_data = {ds: {'fprs': [], 'tprs': [], 'thresholds': []} for ds in df['dataset'].unique()}

for file in results_files:
    with open(file) as f:
        data = json.load(f)
        roc_data[data['dataset']]['fprs'].append(data['roc_curve_data']['fpr'])
        roc_data[data['dataset']]['tprs'].append(data['roc_curve_data']['tpr'])

roc_aggregate = {}
for ds, values in roc_data.items():
    mean_fpr = np.mean(values['fprs'], axis=0)
    mean_tpr = np.mean(values['tprs'], axis=0)
    std_tpr = np.std(values['tprs'], axis=0)
    lower = mean_tpr - std_tpr
    upper = mean_tpr + std_tpr
    roc_aggregate[ds] = {
        'mean_fpr': mean_fpr,
        'mean_tpr': mean_tpr,
        'std_tpr': std_tpr,
        'lower': lower,
        'upper': upper
    }

# Display Metrics Summary
print(metrics_summary)
print(metrics_with_ci)

# Example: Plotting aggregated ROC curve for a dataset
for ds, roc in roc_aggregate.items():
    plt.plot(roc['mean_fpr'], roc['mean_tpr'], color='b', label=f"Mean ROC ({ds})")
    plt.fill_between(roc['mean_fpr'], roc['lower'], roc['upper'], color='b', alpha=0.2, label=f"ROC Std Dev ({ds})")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve for Dataset {ds}")
    plt.legend(loc="lower right")
    plt.show()
