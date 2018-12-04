import numpy as np
from sklearn.metrics import classification_report, confusion_matrix


def mIou(y_true, y_pred, n_classes):
    """
    Mean Intersect over Union metric.
    Computes the one versus all IoU for each class and returns the average.
    Classes that do not appear in the provided set are not counted in the average
    """
    iou = 0
    n_observed = n_classes
    for i in range(n_observed):
        y_t = (y_true == i).astype(int)
        y_p = (y_pred == i).astype(int)

        inter = np.sum(y_t * y_p)
        union = np.sum((y_t + y_p > 0).astype(int))

        if union == 0:
            n_observed -= 1
        else:
            iou += inter / union

    return iou / n_observed


def per_class_IoU(y_true, y_pred, label):
    y_t = (y_true == label).astype(int)
    y_p = (y_pred == label).astype(int)

    inter = np.sum(y_t * y_p)
    union = np.sum((y_t + y_p > 0).astype(int))

    if union == 0:
        return np.nan
    else:
        return inter / union


def per_class_performance(y_true, y_pred):
    perf = classification_report(y_true, y_pred, digits=3, output_dict=True)

    for label, d in perf.items():
        d.update({'IoU': per_class_IoU(y_true, y_pred, label)})

    return d

def conf_mat(y_true,y_pred):
    return confusion_matrix(y_true,y_pred)
