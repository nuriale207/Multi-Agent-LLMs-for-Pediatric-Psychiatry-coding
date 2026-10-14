import numpy as np
import torch
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from transformers import EvalPrediction

def multi_label_metrics(predictions, labels, threshold=0.5):
    # first, apply sigmoid on predictions which are of shape (batch_size, num_labels)
    sigmoid = torch.nn.Sigmoid()
    probs = sigmoid(torch.Tensor(predictions))
    # next, use threshold to turn them into integer predictions
    y_pred = np.zeros(probs.shape)
    y_pred[np.where(probs >= threshold)] = 1
    # finally, compute metrics
    y_true = labels
    metrics=multi_label_metrics_binary(y_true, y_pred)
    print(metrics)
    return metrics

def multi_label_metrics_binary(y_true,y_pred):
    precision_micro_average = precision_score(y_true=y_true, y_pred=y_pred, average='micro')
    precision_macro_average = precision_score(y_true=y_true, y_pred=y_pred, average='macro')
    precision_weighted_average = precision_score(y_true=y_true, y_pred=y_pred, average='weighted')
    precision_samples_average = precision_score(y_true=y_true, y_pred=y_pred, average='samples')
    precision_per_class = precision_score(y_true=y_true, y_pred=y_pred, average=None)

    recall_micro_average = recall_score(y_true=y_true, y_pred=y_pred, average='micro')
    recall_macro_average = recall_score(y_true=y_true, y_pred=y_pred, average='macro')
    recall_weighted_average = recall_score(y_true=y_true, y_pred=y_pred, average='weighted')
    recall_samples_average = recall_score(y_true=y_true, y_pred=y_pred, average='samples')
    recall_per_class = recall_score(y_true=y_true, y_pred=y_pred, average=None)

    f1_micro_average = f1_score(y_true=y_true, y_pred=y_pred, average='micro')
    f1_macro_average = f1_score(y_true=y_true, y_pred=y_pred, average='macro')
    f1_weighted_average = f1_score(y_true=y_true, y_pred=y_pred, average='weighted')
    f1_samples_average = f1_score(y_true=y_true, y_pred=y_pred, average='samples')
    f1_per_class = f1_score(y_true=y_true, y_pred=y_pred, average=None)

    # In an article they propose
    optimized_f1 = np.mean(f1_per_class[np.where(f1_per_class > 0)])
    optimized_recall = np.mean(recall_per_class[np.where(recall_per_class > 0)])
    optimized_precision = np.mean(precision_per_class[np.where(precision_per_class > 0)])

    accuracy = accuracy_score(y_true, y_pred)

    # return a dictionary with all metrics
    metrics = {'precision_micro': precision_micro_average,
               'precision_macro': precision_macro_average,
               'optimized_macro_precision': optimized_precision,
               'precision_weighted': precision_weighted_average,
               'precision_samples': precision_samples_average,

               'recall_micro': recall_micro_average,
               'recall_macro': recall_macro_average,
               'optimized_macro_recall': optimized_recall,

               'recall_weighted': recall_weighted_average,
               'recall_samples': recall_samples_average,
               'f1_micro': f1_micro_average,
               'f1_macro': f1_macro_average,
               'f1_weighted': f1_weighted_average,
               'f1_samples': f1_samples_average,
               'optimized_macro_f1': optimized_f1,
               'accuracy': accuracy}
    return metrics

def compute_metrics(p: EvalPrediction):
    preds = p.predictions[0] if isinstance(p.predictions,
        tuple) else p.predictions
    result = multi_label_metrics(
        predictions=preds,
        labels=p.label_ids)
    print(result)
    return result
