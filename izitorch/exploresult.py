import os
import json
import pickle as pkl
import pandas as pd
import numpy as np


class Fold:
    def __init__(self, folder):
        self.folder = folder
        self.name = self._parse_name()
        self.trainlog_path = os.path.join(folder, 'trainlog.json')
        self.per_class_path = os.path.join(folder, 'per_class_metrics.json')
        self.conf = self._parse_conf()
        self.trainlog = self._parse_trainlog()
        self.per_class = self._parse_per_class()
        self.conf_mat = self._parse_conf_mat()

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
        with open(os.path.join(par_path, 'conf.json'))as src:
            c = json.loads(src.read())
        return c

    def _parse_per_class(self):
        with open(self.per_class_path, 'r') as src:
            t = json.loads(src.read())
        df = pd.DataFrame(t).transpose()

        df = df.drop(['micro avg', 'macro avg', 'weighted avg'], axis=0)

        self.observed_classes = list(map(int, df.index))
        df.index = self.observed_classes

        comp = pd.DataFrame(index=range(self.conf['num_classes']), columns=df.columns)
        for i in range(self.conf['num_classes']):
            if i in df.index:
                comp.iloc[i, :] = df.loc[i]
        return comp.fillna(0)

    def _parse_conf_mat(self):
        return pkl.load(open(os.path.join(self.folder, 'confusion_matrix.pkl'), 'rb'))

    def get_series(self, columns):
        return self.trainlog[columns]

    def get_per_class(self, columns=None):

        if columns is None:
            df = self.per_class
        else:
            df = self.per_class[columns]
        return df

    def get_final_perf(self, columns=None):
        r = self.trainlog.interpolate().iloc[-1, :]
        if columns is None:
            return r
        else:
            return r[columns]


class ModelResult:

    def __init__(self, folder):
        self.folder = folder
        self.name = self._parse_name()
        self.conf = self._parse_conf()
        self.fold_folders = self._get_fold_folders()
        self.nfolds = len(self.fold_folders)
        self.folds = [Fold(f) for f in self.fold_folders]

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
        with open(os.path.join(par_path,'conf.json'))as src:
            c = json.loads(src.read())
        return c

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

    def get_scores(self):
        df = pd.DataFrame()
        for f in self.folds:
            df[f.name] = f.get_final_perf()
        df = df.transpose()
        res = pd.DataFrame()
        res['mean score'] = df.mean()
        res['score std'] = df.std()
        return res.transpose()

    def get_per_class(self, columns=None):
        res = []
        for f in self.folds:
            res.append(f.get_per_class(columns).values)
        df = self.folds[0].get_per_class(columns)
        m = pd.DataFrame(index=df.index, columns=df.columns, data=np.mean(res, axis=0))
        s = pd.DataFrame(index=df.index, columns=df.columns, data=np.std(res, axis=0))

        return m, s

    def get_conf_mat(self):
        return None  # TODO when corrected bug in izitorch (conf mat does not include unobserved labels)
        res = []
        for f in self.folds:
            res.append(f.conf_mat)
        return np.mean(res, axis=0)


class SetupResult:

    def __init__(self, folder):
        self.folder = folder
        self.name = self._parse_name()
        self.models = [ModelResult(f) for f in self._get_model_folders()]

    def _parse_name(self):
        if self.folder[-1] == '/':
            return os.path.split(self.folder[:-1])[-1]
        else:
            return os.path.split(self.folder)[-1]

    def _get_model_folders(self):
        return [os.path.join(self.folder, f) for f in os.listdir(self.folder) if
                os.path.isdir(os.path.join(self.folder, f))]

    def get_series(self, folds=None, columns=['accuracy']):
        res = pd.DataFrame()
        if folds is None:
            folds = list(range(self.models[0].nfolds))

        for m in self.models:
            df = m.get_series(folds=folds, columns=columns)
            df.columns = ['{}_{}'.format(m.name, c) for c in df.columns]
            res = pd.concat([res, df], axis=1)
        return res

    def compare(self):
        res = pd.DataFrame()
        for m in self.models:
            scores = m.get_scores().loc['mean score']
            res[m.name] = scores

        return res.transpose()


class ExpResult:
    def __init__(self, folder):
        self.folder = folder
        self.name = self._parse_name()
        self.setups = [SetupResult(f) for f in self._get_setup_folders()]

    def _parse_name(self):
        if self.folder[-1] == '/':
            return os.path.split(self.folder[:-1])[-1]
        else:
            return os.path.split(self.folder)[-1]

    def _get_setup_folders(self):
        return [os.path.join(self.folder, f) for f in os.listdir(self.folder) if
                os.path.isdir(os.path.join(self.folder, f))]

    def compare(self, metric='test_IoU'):
        res = pd.DataFrame()
        for sr in self.setups:
            res[sr.name] = sr.compare()[metric]
        return res
