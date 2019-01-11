import torch
from torch.utils import data
import torchnet as tnt
import numpy as np
from sklearn import model_selection

from izitorch.metrics import mIou, conf_mat, per_class_performance
from izitorch.utils import weight_init, get_nparams

import argparse
import time
import json
import pickle as pkl
import os


# TODO update docstring
# TODO declare all atributes in init
# TODO optimise redundant blocks
# TODO correct elapsed time display


class Rack:

    def __init__(self):
        torch.manual_seed(1)
        self.model_configs = {}
        self.set_basic_menu()
        self.set_device()
        self.dataset = None

    def to_dict(self):
        output = {}
        for model_name, conf in self.model_configs.items():
            d = vars(self.args).copy()
            d['device'] = str(self.device)

            d['model'] = str(conf['model'])
            d['criterion'] = str(conf['criterion'])
            d['optimizer'] = str(conf['optimizer'])
            d['nparams'] = get_nparams(conf['model'])

            output[model_name] = d
        return output

    ####### Methods for the options menu
    def set_basic_menu(self):
        """
        Sets a basic argparse menu with the common arguments
        """
        parser = argparse.ArgumentParser()

        parser.add_argument('--dataset', default='/home/vsfg/data')
        parser.add_argument('--num_classes', default=None, type=int,
                            help='number of classes in case of classification problem')

        parser.add_argument('--res_dir', default='results', help='folder for saving the trained model')
        parser.add_argument('--resume', default='')
        parser.add_argument('--save_all', default=0, type=int,
                            help='If 0, will save only weigths of the last testing step.'
                                 'If 1, will save the weights of all testing steps.')
        parser.add_argument('--validation', default=0, type=int,
                            help='If set to 1 each epoch will be tested on a validation set,'
                                 ' and the best epoch will be used for the final test on a separate test set')
        parser.add_argument('--batch_size', default=128, type=int, help='Batch size')
        parser.add_argument('--shuffle', default=True, help='Shuffle dataset')
        parser.add_argument('--grad_clip', default=0, type=float,
                            help='If nonzero, absolute balue of the gradients will be clipped at this value')
        parser.add_argument('--num_workers', default=6, type=int, help='number of workers for data loader')
        parser.add_argument('--train_ratio', default=.8, type=float, help='ratio for train/test split')
        parser.add_argument('--kfold', default=0, type=int,
                            help='If non zero, number of folds for KFCV, and overwrites train_ratio argument')
        parser.add_argument('--pin_memory', default=0, type=int, help='whether to use pin_memory for dataloader')

        parser.add_argument('--epochs', default=1000, type=int)
        parser.add_argument('--lr', default=1e-3, type=float, help='Initial learning rate')
        parser.add_argument('--lr_decay', default=1, type=float,
                            help='Multiplicative factor used on learning rate at lr_steps')
        parser.add_argument('--lr_steps', default='',
                            help='List of epochs where the learning rate is decreased by `lr_decay`, separate with a +')
        parser.add_argument('--test_step', default=10, type=int, help='Test model every that many steps')

        self.parser = parser

    def add_arguments(self, arg_dict):
        """
        Add custom arguments to the argparse menu
        Args:
            arg_dict (dict): {arg_name:{'default':default_value,'type':arg_type}
        """
        for name, setting in arg_dict.items():
            self.parser.add_argument('--{}'.format(name), default=setting['default'], type=setting['type'])

    def parse_args(self):
        self.args = self.parser.parse_args()

    def print_args(self):
        print(self.args)

    def _check_args_consistency(self):
        if self.args.validation and (self.args.test_step == 0 or self.args.save_all == 0):
            print('[WARNING] Validation requires testing at each epoch, setting test step and save all to 1')
            self.args.test_step = 1
            self.args.save_all = 1
        if self.args.kfold < 3 and self.args.validation:
            print('[WARNING] K-fold training with validation requires k > 2, setting k=3')
            self.args.kfold = 3

    ####### Methods for setting the specific elements of the training rack

    def set_device(self, device_name=None):
        """Sets the device used by torch for computation, will prioritize GPU by default but can be set manually"""
        try:
            self.device = torch.device(device_name)
        except TypeError:
            self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    def add_model_configs(self, model_configs):
        """
        Add model configurations to be trainined in the current script
        Args:
            model_configs (dict): model configuration with entries in the form:
                        {
                            model_name:{
                                        'model':torch.nn instance,
                                        'optimizer': torch optimizer instance,
                                        'criterion' : torch.nn criterion instance,
                                        'scheduler' : (optional) torch.optim learning rate scheduler
                            }
                        }
        Returns:

        """
        self.model_configs.update(model_configs)

    def set_dataset(self, dataset):
        """
        Attaches a dataset to the training rack. If only one dataset is provided, it will be split in train and test,
        but a list of two datasets can also be provided (allows to have different image transformation on train and test).
        For generality's sake the dataset should return items in the form (input,target).
        If the network uses more exotic input, this needs to be dealt with in the forward method of its definition
        Args:
            dataset: instance of torch.utils.dataset or list [train_dataset, test_dataset]
        """
        if isinstance(dataset, torch.utils.data.Dataset):
            self.dataset = dataset
        elif isinstance(dataset, list):
            self.train_dataset, self.test_dataset = dataset
        else:
            raise ValueError

    ####### Methods for preparation

    def _prepare_output(self):
        """
        Creates output directory and writes the configuration file in it.

        """

        repr = self.to_dict()

        for model_name in self.model_configs:

            res_dir = os.path.join(self.args.res_dir, model_name)
            self.model_configs[model_name]['res_dir'] = res_dir

            if not os.path.exists(res_dir):
                os.makedirs(res_dir)
            else:
                print('[WARNING] Output directory  already exists')

            if self.args.kfold != 0:
                for i in range(self.args.kfold):
                    os.makedirs(os.path.join(res_dir, 'FOLD_{}'.format(i + 1)), exist_ok=True)

            with open(os.path.join(res_dir, 'conf.json'), 'w') as fp:
                json.dump(repr[model_name], fp, indent=4)

    def _get_loaders(self):
        """
        Splits the dataset in train and test and returns a list of train and test dataloader pairs.
        Each pair of dataloader of the list corresponds to one fold in case of k-fold, and the list is of length one if
        there is no k-fold
        """
        if self.args.pin_memory == 1:
            pm = True
        else:
            pm = False

        if self.dataset is not None:
            self.train_dataset = self.dataset
            self.test_dataset = self.dataset

        print('[DATASET] Splitting dataset')

        indices = list(range(len(self.train_dataset)))
        if self.args.shuffle:
            np.random.seed(128)  # TODO random seed as an option
            np.random.shuffle(indices)

        if self.args.kfold != 0:
            kf = model_selection.KFold(n_splits=self.args.kfold, random_state=1, shuffle=False)
            indices_seq = list(kf.split(list(range(len(indices)))))
            self.ntrain = len(indices_seq[0][0])
            self.ntest = len(indices_seq[0][1])
            print('[TRAINING CONFIGURATION] Preparing {}-fold cross validation'.format(self.args.kfold))

        else:
            self.ntrain = int(np.floor(self.args.train_ratio * len(self.train_dataset)))
            self.ntest = len(self.train_dataset) - self.ntrain
            indices_seq = [(list(range(self.ntrain)), list(range(self.ntrain, self.ntrain + self.ntest, 1)))]

            ####### TODO
            print('[DATASET] Train: {} samples, Test : {} samples'.format(self.ntrain, self.ntest))

        loader_seq = []

        record = []
        for train, test in indices_seq:
            train_indices = [indices[i] for i in train]
            test_indices = [indices[i] for i in test]

            if self.args.validation:
                validation_indices = np.random.choice(train_indices, size=self.ntest, replace=False)
                train_indices = [t for t in train_indices if
                                 t not in validation_indices]  # TODO Find a less expensive way to do this

                record.append((train_indices, validation_indices, test_indices))

                train_sampler = data.sampler.SubsetRandomSampler(train_indices)
                validation_sampler = data.sampler.SubsetRandomSampler(validation_indices)
                test_sampler = data.sampler.SubsetRandomSampler(test_indices)

                train_loader = data.DataLoader(self.train_dataset, batch_size=self.args.batch_size,
                                               sampler=train_sampler,
                                               num_workers=self.args.num_workers, pin_memory=pm)
                validation_loader = data.DataLoader(self.train_dataset, batch_size=self.args.batch_size,
                                                    sampler=validation_sampler,
                                                    num_workers=self.args.num_workers, pin_memory=pm)
                test_loader = data.DataLoader(self.test_dataset, batch_size=self.args.batch_size,
                                              sampler=test_sampler,
                                              num_workers=self.args.num_workers, pin_memory=pm)
                loader_seq.append((train_loader, validation_loader, test_loader))

            else:

                record.append((train_indices, test_indices))

                train_sampler = data.sampler.SubsetRandomSampler(train_indices)
                test_sampler = data.sampler.SubsetRandomSampler(test_indices)

                train_loader = data.DataLoader(self.train_dataset, batch_size=self.args.batch_size,
                                               sampler=train_sampler,
                                               num_workers=self.args.num_workers, pin_memory=pm)
                test_loader = data.DataLoader(self.test_dataset, batch_size=self.args.batch_size,
                                              sampler=test_sampler,
                                              num_workers=self.args.num_workers, pin_memory=pm)
                loader_seq.append((train_loader, test_loader))

        pkl.dump(record, open(os.path.join(self.args.res_dir, 'Dataset_split.pkl'), 'wb'))

        return loader_seq

    def _init_weights(self):
        for conf in self.model_configs.values():
            conf['model'] = conf['model'].apply(weight_init)

    ####### Methods for execution

    def launch(self):  # TODO
        """
        MAIN METHOD: does the necessary preparations and launches the training loop for the specified number of epochs
        the model is applied to the test dataset every args.test_step epochs
        """

        self._check_args_consistency()
        self._prepare_output()

        loader_seq = self._get_loaders()
        nfold = len(loader_seq)

        for i, loaders in enumerate(loader_seq):

            if self.args.validation:
                self.train_loader, self.validation_loader, self.test_loader = loaders
            else:
                self.train_loader, self.test_loader = loaders

            if nfold == 1:
                print('[TRAINING CONFIGURATION] Starting single training ')
                subdir = ''
            else:
                print('[TRAINING CONFIGURATION] Training with {}-fold cross validation'.format(nfold))
                subdir = 'FOLD_{}'.format(i + 1)

            self.args.total_step = len(self.train_loader)

            self._init_weights()

            self._init_trainlogs()

            self._models_to_device()

            for self.current_epoch in range(self.args.epochs):
                t0 = time.time()

                train_metrics = self.train_epoch()
                self.checkpoint_epoch(self.current_epoch, train_metrics, subdir=subdir)

                t1 = time.time()

                print('[PROGRESS] Epoch duration : {}'.format(t1 - t0))
                print('####################################################')

    def _init_trainlogs(self):
        self.stats = {}
        self.best_performance = {}
        for model_name, conf in self.model_configs.items():
            self.stats[model_name] = {}
            self.best_performance[model_name] = {'epoch': 0, 'IoU': 0}

    def _models_to_device(self):
        for model_name, conf in self.model_configs.items():
            conf['model'] = conf['model'].to(self.device)

    def train_epoch(self):  # TODO
        """
        Trains the model on one epoch and displays the evolution of the training metrics.
        Returns a dictionary with the performance metrics over the whole epoch.
        """
        acc_meter = {}
        loss_meter = {}

        y_true = []
        y_pred = {m: [] for m in self.model_configs}

        for model_name, conf in self.model_configs.items():
            conf['model'] = conf['model'].train()
            acc_meter[model_name] = tnt.meter.ClassErrorMeter(accuracy=True)
            loss_meter[model_name] = tnt.meter.AverageValueMeter()

        ta = time.time()

        for i, (x, y) in enumerate(self.train_loader):

            try:
                x = x.to(self.device)
            except AttributeError:  # dirty fix for extra data
                for j, input in enumerate(x):
                    x[j] = input.to(self.device)

            y_true.extend(list(map(int, y)))
            y = y.to(self.device)

            loss = {}
            outputs = {}
            prediction = {}

            for model_name, conf in self.model_configs.items():
                outputs[model_name] = conf['model'](x)
                loss[model_name] = conf['criterion'](outputs[model_name], y.long())
                prediction[model_name] = outputs[model_name].detach()

                acc_meter[model_name].add(prediction[model_name], y)
                loss_meter[model_name].add(loss[model_name].item())

                y_p = prediction[model_name].argmax(dim=1).cpu().numpy()
                y_pred[model_name].extend(list(y_p))

                conf['optimizer'].zero_grad()
                loss[model_name].backward()

                if self.args.grad_clip > 0:
                    for p in conf['model'].parameters():
                        p.grad.data.clamp_(-self.args.grad_clip, self.args.grad_clip)

                conf['optimizer'].step()

                if 'scheduler' in conf:
                    conf['scheduler'].step()  # TODO there is apparently a bug when scheduler is used, to be fixed

                if (i + 1) % 100 == 0:
                    tb = time.time()
                    elapsed = tb - ta
                    print('[{:20.20}] Step [{}/{}], Loss: {:.4f}, Acc : {:.2f}, Duration:{:.4f}'
                          .format(model_name, i + 1, self.args.total_step, loss_meter[model_name].value()[0],
                                  acc_meter[model_name].value()[0],
                                  elapsed))
                    ta = tb

        metrics = {}
        for model_name in self.model_configs:
            miou = mIou(y_true, y_pred[model_name], self.args.num_classes)
            metrics[model_name] = {'loss': loss_meter[model_name].value()[0],
                                   'accuracy': acc_meter[model_name].value()[0],
                                   'IoU': miou}

        return metrics

    def checkpoint_epoch(self, epoch, metrics, subdir=''):  # TODO make it cleaner
        """
        Computes accuracy and loss of the epoch and writes it in the trainlog, along with the model weights.
        If on a test epoch the test metrics will also be computed and writen in the trainlog
        Args:
            epoch (int): number of the epoch
            metrics (dict): training metrics to be written
            subdir (str): Optional, name of the target sub directory (used for k-fold)
        """

        if epoch % self.args.test_step == 0 or epoch + 1 == self.args.epochs:

            if epoch + 1 == self.args.epochs:
                test_metrics, (y_true, y_pred) = self.test(return_y=True)
            else:
                test_metrics = self.test()

            for model_name, conf in self.model_configs.items():
                self.stats[model_name][epoch + 1] = {**metrics[model_name], **test_metrics[model_name]}  # TODO
                print('[PROGRESS - {}] Writing checkpoint of epoch {}\{} . . .'.format(model_name, epoch + 1,
                                                                                       self.args.epochs))

                with open(os.path.join(conf['res_dir'], subdir, 'trainlog.json'), 'w') as outfile:
                    json.dump(self.stats[model_name], outfile, indent=4)
                if self.args.save_all == 1:
                    file_name = 'model_epoch{}.pth.tar'.format(epoch + 1)
                else:
                    file_name = 'model.pth.tar'

                torch.save({'epoch': epoch + 1, 'state_dict': conf['model'].state_dict(),
                            'optimizer': conf['optimizer'].state_dict()},
                           os.path.join(conf['res_dir'], subdir, file_name))

            if epoch + 1 == self.args.epochs:

                if self.args.validation:
                    y_true, y_pred = self.get_best_predictions(subdir=subdir)
                for model_name, conf in self.model_configs.items():
                    per_class, conf_m = self.final_performance(y_true, y_pred[model_name],self.args.num_classes)
                    with open(os.path.join(conf['res_dir'], subdir, 'per_class_metrics.json'), 'w') as outfile:
                        json.dump(per_class, outfile, indent=4)
                    pkl.dump(conf_m, open(os.path.join(conf['res_dir'], subdir, 'confusion_matrix.pkl'), 'wb'))
        else:
            for model_name, conf in self.model_configs.items():
                self.stats[model_name][epoch + 1] = metrics[model_name]
                print('[PROGRESS - {}] Writing checkpoint of epoch {}\{} . . .'.format(model_name, epoch + 1,
                                                                                       self.args.epochs))

                with open(os.path.join(conf['res_dir'], subdir, 'trainlog.json'), 'w') as outfile:
                    json.dump(self.stats[model_name], outfile, indent=4)

    def test(self, return_y=False):  # TODO
        """
        Tests the model on the test or validation set and returns the performance metrics as a dictionnary
        """
        # TODO generalise method and just give a list of metrics as rack parameter
        print('Testing models . . .')

        acc_meter = {}
        loss_meter = {}

        y_true = []
        y_pred = {m: [] for m in self.model_configs}

        for model_name, conf in self.model_configs.items():
            conf['model'] = conf['model'].eval()

            acc_meter[model_name] = tnt.meter.ClassErrorMeter(accuracy=True)
            loss_meter[model_name] = tnt.meter.AverageValueMeter()

        loader = self.validation_loader if self.args.validation else self.test_loader

        for (x, y) in loader:

            y_true.extend(list(map(int, y)))

            x = x.to(self.device)
            y = y.to(self.device)

            prediction = {}
            loss = {}
            for model_name, conf in self.model_configs.items():
                with torch.no_grad():
                    prediction[model_name] = conf['model'](x)
                    loss[model_name] = conf['criterion'](prediction[model_name], y)

                acc_meter[model_name].add(prediction[model_name], y)
                loss_meter[model_name].add(loss[model_name].item())

                y_p = prediction[model_name].argmax(dim=1).cpu().numpy()
                y_pred[model_name].extend(list(y_p))

        test_metrics = {}
        for model_name in self.model_configs:
            acc = acc_meter[model_name].value()[0]
            loss = loss_meter[model_name].value()[0]
            miou = mIou(y_true, y_pred[model_name], self.args.num_classes)

            test_metrics[model_name] = {'test_accuracy': acc, 'test_loss': loss, 'test_IoU': miou}

            print('[PERFORMANCE - {}] Test accuracy : {:.3f}'.format(model_name, acc))
            print('[PERFORMANCE - {}] Test loss : {:.4f}'.format(model_name, loss))
            print('[PERFORMANCE - {}] Test IoU : {:.4f}'.format(model_name, miou))

            if self.args.validation:
                if test_metrics[model_name]['test_IoU'] > self.best_performance[model_name]['IoU']:
                    print('[PERFORMANCE - {}] BEST EPOCH !'.format(model_name))
                    self.best_performance[model_name]['IoU'] = test_metrics[model_name]['test_IoU']
                    self.best_performance[model_name]['epoch'] = self.current_epoch + 1

                test_metrics[model_name] = {'val_accuracy': acc, 'val_loss': loss, 'val_IoU': miou}

        if return_y:
            return test_metrics, (y_true, y_pred)
        else:
            return test_metrics

    def final_test(self):
        acc_meter = {}
        loss_meter = {}

        y_true = []
        y_pred = {m: [] for m in self.model_configs}

        for model_name, conf in self.model_configs.items():

            if self.args.validation:
                print('[TESTING - {}')
            conf['model'] = conf['model'].eval()
            acc_meter[model_name] = tnt.meter.ClassErrorMeter(accuracy=True)
            loss_meter[model_name] = tnt.meter.AverageValueMeter()

        for (x, y) in self.test_loader:
            y_true.extend(list(map(int, y)))

            x = x.to(self.device)
            y = y.to(self.device)

            for model_name, conf in self.model_configs.items():
                with torch.no_grad():
                    prediction = conf['model'](x)
                    loss = conf['criterion'](prediction, y)

                acc_meter[model_name].add(prediction, y)
                loss_meter[model_name].add(loss.item())

                y_p = prediction.argmax(dim=1).cpu().numpy()
                y_pred[model_name].extend(list(y_p))

        test_metrics = {}
        per_class = {}
        conf_m = {}

        for model_name in self.model_configs:
            acc = acc_meter[model_name].value()[0]
            loss = loss_meter[model_name].value()[0]
            miou = mIou(y_true, y_pred[model_name], self.args.num_classes)

            per_class[model_name] = per_class_performance(y_true, y_pred[model_name], self.args.num_classes)
            conf_m[model_name] = conf_mat(y_true, y_pred[model_name],self.args.num_classes)

            test_metrics[model_name] = {'test_accuracy': acc, 'test_loss': loss, 'test_IoU': miou}

            print('[PERFORMANCE - {}] Test accuracy : {:.3f}'.format(model_name, acc))
            print('[PERFORMANCE - {}] Test loss : {:.4f}'.format(model_name, loss))
            print('[PERFORMANCE - {}] Test IoU : {:.4f}'.format(model_name, miou))

        return test_metrics, per_class, conf_m

    def final_performance(self, y_true, y_pred):

        per_class = per_class_performance(y_true, y_pred,self.args.num_classes)
        conf_m = conf_mat(y_true, y_pred,self.args.num_classes)

        return per_class, conf_m

    def get_best_predictions(self, subdir=''):

        y_true = []
        y_pred = {m: [] for m in self.model_configs}

        for model_name, conf in self.model_configs.items():
            best_epoch = self.best_performance[model_name]['epoch']
            checkpoint = torch.load(os.path.join(conf['res_dir'], subdir, 'model_epoch{}.pth.tar'.format(
                best_epoch)))  ## TODO if validation then save_all
            conf['model'].load_state_dict(checkpoint['state_dict'])
            conf['model'].eval()

        for (x, y) in self.test_loader:

            y_true.extend(list(map(int, y)))
            x = x.to(self.device)

            for model_name, conf in self.model_configs.items():
                with torch.no_grad():
                    prediction = conf['model'](x)
                    y_pred[model_name].extend(prediction.argmax(dim=1).cpu().numpy())
        return y_true, y_pred
