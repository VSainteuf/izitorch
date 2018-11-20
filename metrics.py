import numpy as np


def mIou(y_true, y_pred, n_classes):
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
