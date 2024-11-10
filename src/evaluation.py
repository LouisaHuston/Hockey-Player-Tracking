# src/evaluation.py
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

def calculate_accuracy(y_true, y_pred):
    """
    Calculate accuracy.
    
    :param y_true: List or array of true labels
    :param y_pred: List or array of predicted labels
    :return: Accuracy score
    """
    return accuracy_score(y_true, y_pred)

def calculate_precision(y_true, y_pred):
    """
    Calculate precision.
    
    :param y_true: List or array of true labels
    :param y_pred: List or array of predicted labels
    :return: Precision score
    """
    return precision_score(y_true, y_pred, average='weighted', zero_division=1)

def calculate_recall(y_true, y_pred):
    """
    Calculate recall.
    
    :param y_true: List or array of true labels
    :param y_pred: List or array of predicted labels
    :return: Recall score
    """
    return recall_score(y_true, y_pred, average='weighted', zero_division=1)

def calculate_f1_score(y_true, y_pred):
    """
    Calculate F1 score.
    
    :param y_true: List or array of true labels
    :param y_pred: List or array of predicted labels
    :return: F1 score
    """
    return f1_score(y_true, y_pred, average='weighted', zero_division=1)
