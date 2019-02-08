"""
Set of helper classes to parse and explore the checkpoints and trainlogs of the train rack, in case of a classification
problem.
"""

import os
import json
import pickle as pkl
import pandas as pd
import numpy as np


class FoldResult:
    def __init__(self, folder):
        """
        Initialise and parse the trainlog, confusion_matrix and configuration of the fold.
        Also directly computes the final performance achieved.

        Args:
            folder (str): path to the directory of the fold's outputs
        """
        self.folder = folder
        self.name = self._parse_name()
        self.trainlog_path = os.path.join(folder, 'trainlog.json')
        self.conf = self._parse_conf()
        self.trainlog = self._parse_trainlog()
        self.conf_mat = self._parse_conf_mat()
        self.per_class, self.overall = self._compute_metrics()

    def _parse_name(self):
        if self.folder[-1] == '/':
            return os.path.split(self.folder[:-1])[-1]
        else:
            return os.path.split(self.folder)[-1]

    def _parse_trainlog(self):
        with open(self.trainlog_path, 'r') as src:
            t = json.loads(src.read())
        df = pd.DataFrame(t).transpose()
        for c in [c for c in df.columns if 'IoU' in c]:
            df[c] = df[c] * 100

        return df

    def _is_kfold(self):
        return 'FOLD_' in self.folder

    def _with_val(self):
        return bool(self.conf['validation'])

    def _parse_conf(self):
        if self._is_kfold():
            par_path = os.path.abspath(os.path.join(self.folder, '..'))
        else:
            par_path = self.folder
        with open(os.path.join(par_path, 'conf.json')) as src:
            c = json.loads(src.read())
        return c

    def _parse_conf_mat(self):
        return pkl.load(open(os.path.join(self.folder, 'confusion_matrix.pkl'), 'rb'))

    def _compute_metrics(self):
        return metrics(self.conf_mat)

    def get_series(self, columns):
        """
        Gets the specified set of metrics from the trainlog.
        Args:
            columns (list): names of the columns to be retrieved (e.g. 'accuracy', 'loss', 'val_accuracy')

        Returns:
            df (pandas.DataFrame) :  DataFrame with the values of the selected metrics for all the epochs
        """
        return self.trainlog[columns]

    def get_confmat(self):
        """Gets the confusion matrix
        """
        return self.conf_mat

    def get_per_class(self):
        """Gets the per_class performance"""
        return self.per_class

    def get_overall(self):
        """Gets the overall performance"""
        return self.overall


class ModelResult:

    def __init__(self, folder):
        """
          Initialise and parse the trainlog, confusion_matrix and configuration of the model.
          If the model was trained with K-fold, all the folds are parsed and combined to compute performance.

          Args:
              folder (str): path to the directory of the model's outputs
          """
        self.folder = folder
        self.name = self._parse_name()
        self.conf = self._parse_conf()
        self.fold_folders = self._get_fold_folders()
        self.nfolds = len(self.fold_folders)
        self.folds = [FoldResult(f) for f in self.fold_folders]
        self.conf_mat = self._assemble_conf_mat()
        self.per_class, self.overall = self._compute_metrics()

    def __repr__(self):
        return self.name

    def _parse_name(self):
        if self.folder[-1] == '/':
            return os.path.split(self.folder[:-1])[-1]
        else:
            return os.path.split(self.folder)[-1]

    def _get_fold_folders(self):
        folds = [os.path.join(self.folder, f) for f in os.listdir(self.folder) if
                 'FOLD_' in f and os.path.isdir(os.path.join(self.folder, f))]
        if len(folds) == 0:
            return [self.folder]
        else:
            return folds

    def _parse_conf(self):
        par_path = os.path.abspath(self.folder)
        with open(os.path.join(par_path, 'conf.json'))as src:
            c = json.loads(src.read())
        return c

    def _assemble_conf_mat(self):
        return sum([f.conf_mat for f in self.folds])

    def _compute_metrics(self):
        return metrics(self.conf_mat)

    def get_nparams(self):
        return self.conf['nparams']

    def get_series(self, folds=None, columns=None):
        res = pd.DataFrame()
        if folds is None:
            folds = list(range(self.nfolds))

        for f in [self.folds[i] for i in folds]:
            df = f.get_series(columns)
            df.columns = ['{}_{}'.format(f.name, c) for c in columns]
            res = pd.concat([res, df], axis=1)

        return res

    def get_confmat(self):
        return self.conf_mat

    def get_per_class(self):
        return self.per_class

    def get_overall(self):
        return self.overall


class Comparator:

    def __init__(self, model_folder_list):
        """
        Helper class to compare the results of different models

        Args:
            model_folder_list (list): list of the paths to the models that should be compared
        """
        self.folders = model_folder_list
        self.models = [ModelResult(f) for f in model_folder_list]

    def overall(self, metrics=['Precision', 'Recall', 'F1-score', 'IoU'], mode='MACRO'):
        """
        Assembles the overall performances of the different models in a single table.
        Args:
            metrics (list): list of the metrics that should be used for comparison
            mode (str): which of MICRO or MACRO averages to choose to aggregate the per class metrics

        Returns:
            df (pandas.DataFrame): DataFrame with the specified metrics for all models under consideration
        """
        columns = ['{}_{}'.format(mode, met) for met in metrics if met != 'Accuracy']
        if 'Accuracy' in metrics:
            columns.append('Accuracy')
        df = pd.DataFrame(columns=columns)
        for m in self.models:
            p = []
            for met in columns:
                p.append(m.get_overall()[met])
            df.loc[m.name] = p
        return df

    def per_class(self, metric='F1-score'):
        """
        Assembles the per class performance of all models for a specific metric, in a single table.
        Args:
            metric (str): metrics that should be used for comparison

        Returns:
            df (pandas.DataFrame): DataFrame with the per class performance for all models under consideration.

        """
        df = pd.DataFrame()
        for m in self.models:
            df[m.name] = pd.DataFrame(m.get_per_class()).transpose()[metric]
        return df


def metrics(mat):
    """
    This method computes all the performance metrics from the confusion matrix. In addition to overall accuracy, the
    precision, recall, f-score and IoU for each class is computed.
    The class-wise metrics are averaged to provide overall indicators in two ways (MICRO and MACRO average)
    Args:
        mat (array): confusion matrix

    Returns:
        per_class (dict) : per class metrics
        overall (dict): overall metrics

    """
    TP = 0
    FP = 0
    FN = 0

    per_class = {}

    for j in range(mat.shape[0]):
        d = {}
        tp = np.sum(mat[j, j])
        fp = np.sum(mat[:, j]) - tp
        fn = np.sum(mat[j, :]) - tp

        d['IoU'] = tp / (tp + fp + fn)
        d['Precision'] = tp / (tp + fp)
        d['Recall'] = tp / (tp + fn)
        d['F1-score'] = 2 * tp / (2 * tp + fp + fn)

        per_class[str(j)] = d

        TP += tp
        FP += fp
        FN += fn

    overall = {}
    overall['micro_IoU'] = TP / (TP + FP + FN)
    overall['micro_Precision'] = TP / (TP + FP)
    overall['micro_Recall'] = TP / (TP + FN)
    overall['micro_F1-score'] = 2 * TP / (2 * TP + FP + FN)

    macro = pd.DataFrame(per_class).transpose().mean()
    overall['MACRO_IoU'] = macro.loc['IoU']
    overall['MACRO_Precision'] = macro.loc['Precision']
    overall['MACRO_Recall'] = macro.loc['Recall']
    overall['MACRO_F1-score'] = macro.loc['F1-score']

    overall['Accuracy'] = np.sum(np.diag(mat)) / np.sum(mat)

    return per_class, overall
