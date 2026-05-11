from sklearn.metrics import classification_report, f1_score, roc_auc_score, roc_curve
from scipy.optimize import brentq
from scipy.interpolate import interp1d

"""
Evaluation Metrics for Binary Classification (Fake vs Real Images) using sklearn library:
===================================================================================
1. Classification Report: Provides precision, recall, F1-score, and support for each class.
2. F1-Score: Harmonic mean of Precision and Recall.
3. AUC (Area Under the ROC Curve): Probability that a random "Fake" image will have a higher score than a random "Real" image.
4. EER (Equal Error Rate): The point where False Positive Rate and False Negative Rate are equal.

Inputs and outputs:
    y_test: your true labels (0 or 1)
    final_preds: your model's 0/1 predictions (for F1)
    y_probs: your model's probability scores (for AUC and EER) 
"""

# A classification report using sklearn library, which includes precision, recall, F1-score, and support for each class.
def get_classification_report(y_test, final_preds, digits=4):
    return classification_report(y_test, final_preds, digits= digits)

# This is the harmonic mean of Precision and Recall.
def get_f1_score(y_test, final_preds):
    return f1_score(y_test, final_preds)

# Represents the probability that a random "Fake" image will have a higher score than a random "Real" image.
def get_auc_score(y_test, y_probs):
    return roc_auc_score(y_test, y_probs)

# The point where False Positive Rate and False Negative Rate are equal. Lower is better (0.0 is a perfect model).
def get_eer_score(y_test, y_probs):
    fpr, tpr, thresholds = roc_curve(y_test, y_probs)
    return brentq(lambda x : 1. - x - interp1d(fpr, tpr)(x), 0., 1.)