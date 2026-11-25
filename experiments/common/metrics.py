import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, recall_score, roc_auc_score, make_scorer
METRICS = ['recall', 'f1', 'gmean', 'auc']

def gmean_score(truth, prediction):
    tn, fp, fn, tp = confusion_matrix(truth, prediction, labels=[0, 1]).ravel()
    sensitivity = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    return float(np.sqrt(sensitivity * specificity))
GMEAN_SCORER = make_scorer(gmean_score)

def score_predictions(truth, prediction, scores):
    tn, fp, fn, tp = confusion_matrix(truth, prediction, labels=[0, 1]).ravel()
    sensitivity = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    return dict(recall=recall_score(truth, prediction, zero_division=0), f1=f1_score(truth, prediction, zero_division=0), gmean=float(np.sqrt(sensitivity * specificity)), auc=roc_auc_score(truth, scores))
