import numpy as np
from sklearn.metrics import classification_report, confusion_matrix


def mIou(y_true, y_pred, n_classes):
    """
    Mean Intersect over Union metric.
    Computes the one versus all IoU for each class and returns the average.
    Classes that do not appear in the provided set are not counted in the average.
    Args:
        y_true (1D-array): True labels
        y_pred (1D-array): Predicted labels
        n_classes (int): Total number of classes
    Returns:
        mean Iou (float)
    """
    iou = 0
    n_observed = n_classes
    for i in range(n_classes):
        y_t = (np.array(y_true) == i).astype(int)
        y_p = (np.array(y_pred) == i).astype(int)

        inter = np.sum(y_t * y_p)
        union = np.sum((y_t + y_p > 0).astype(int))

        if union == 0:
            n_observed -= 1
        else:
            iou += inter / union

    return iou / n_observed


def per_class_IoU(y_true, y_pred, label):
    """
    Computes the one versus all IoU for one class
    Args:
        y_true (1D-array): True labels
        y_pred (1D-array): Predicted labels
        label (int): label of the class under consideration

    Returns:
        IoU (float)
    """
    y_t = (np.array(y_true) == label).astype(int)
    y_p = (np.array(y_pred) == label).astype(int)

    inter = np.sum(y_t * y_p)
    union = np.sum((y_t + y_p > 0).astype(int))

    if union == 0:
        return np.nan
    else:
        return inter / union


def per_class_performance(y_true, y_pred, n_classes):
    """
    Computes per-class Accuracy, Precision, Recall, F-score, and IoU
    Args:
        y_true (1D-array): True labels
        y_pred (1D-array): Predicted labels
        n_classes (int): Total number of classes

    Returns:
        perf (dict)

    """
    perf = classification_report(y_true, y_pred, digits=3, output_dict=True, labels=list(range(n_classes)))

    for label, d in perf.items():
        if label not in ['micro avg', 'macro avg', 'weighted avg']:
            d.update({'IoU': per_class_IoU(y_true, y_pred, int(label))})

    return perf


def conf_mat(y_true, y_pred, n_classes):
    """
    Computes the confusion matrix of a classification prediction. The total number of classes is explicitly passed
    as an argument so that it is not assumed from the provided set of labels.
    Args:
        y_true (1D-array): True labels
        y_pred (1D-array): Predicted labels
        n_classes (int): Total number of classes

    Returns:
        confusion matrix (2D-array)

    """
    return confusion_matrix(y_true, y_pred, labels=list(range(n_classes)))
