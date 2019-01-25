"""
Main script of the package. It implements an abstract training rack class which manages common mechanisms of
training models on pytorch: training iterations, back-propagation, training scheme (e.g. k-fold or simple training,
with or without a validation set, testing), dataset splitting, and checkpoints.
The idea is to manage all these aspects under the hood, with a fair degree of flexibility, and to reduce the user's
scope to providing the dataset to be used, and defining the models that are to be trained.
The package was mainly developed for image classification problems but should easily be applied to other types of
problems.
"""

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
import sys


# TODO Add resume training feature
# TODO Add a stop at convergence feature
# TODO Remove per class metrics

class Rack:
    """
    Pytorch model training rack. It implements an argparse menu to set the parameters of the training program, and the
    methods to execute it. The class is meant to be used in a script that is then called in command line (see examples).
    After initialisation, custom arguments can be added to the argparse menu.
    The neccessary steps to launch a training are as follows:
        - Parse the arguments (rack.parse_args())
        - Set the dataset (rack.set_dataset(dataset))
        - Set the model configuration(s) (rack.add_model_configs(configs))
        (the rack supports multi-model training i.e. training multiple models on the same dataset at the same time)
        - Launch the training (rack.launch())

    Attributes:
        parser (argparse.ArgumentParser): Rack argument parser
        args (argparse.ArgumentParser.args): Parameters retrieved from the argparse menu
        device (torch.device): Device used for computations
        dataset_train_transforms (torch.utils.data.Dataset): Dataset used for training
        dataset_test_transforms (torch.utils.data.Dataset): Dataset used for testing ##TODO Clarifiy
        ntrain (int): Number of training samples
        ntest (int): Number of testing samples
        train_loader  (torch.utils.data.DataLoader): Dataloader for training
        test_loader  (torch.utils.data.DataLoader): Dataloader for testing
        validation_loader  (torch.utils.data.DataLoader): Dataloader for validation
        current_epoch (int): Active epoch in training loop
        stats (dict): dictionary containing the trainlogs
        best_performance (dict): dictionary keeping track of the best epoch
        model_configs (list): list of trainRack.ModelConfig instances
    """

    def __init__(self):
        """Initiliazes the rack and attaches a basic argparse menu to it."""
        torch.manual_seed(1)  # TODO random seed as an option
        self.parser = None
        self.args = None
        self.device = None

        self.dataset_train_transforms = None
        self.dataset_test_transforms = None
        self.ntrain = None
        self.ntest = None

        self.train_loader = None
        self.validation_loader = None
        self.test_loader = None
        self.current_epoch = None

        self.stats = None
        self.best_performance = None

        self.model_configs = []
        self._set_basic_menu()

    def to_dict(self):
        """Summarizes the rack's configuration in the form of a dictionary"""
        output = {}

        for mc in self.model_configs:
            d = vars(self.args).copy()
            d['model'] = str(mc.model)
            d['criterion'] = str(mc.criterion)
            d['optimizer'] = str(mc.optimizer)
            d['nparams'] = get_nparams(mc.model)
            output[mc.name] = d
        return output

    ####### Methods for the options menu
    def _set_basic_menu(self):
        """
        Sets a basic argparse menu with commonly used arguments
        """
        parser = argparse.ArgumentParser()

        # Basic arguments
        parser.add_argument('--device', default='cuda', help='device to use for tensor computations (cpu/cuda)')
        parser.add_argument('--res_dir', default='results', help='folder for saving the trained model')
        # parser.add_argument('--resume', default='')  # TODO implement resuming a training

        parser.add_argument('--dataset', default='/home/vsfg/data')
        parser.add_argument('--num_classes', default=None, type=int,
                            help='number of classes in case of classification problem')
        parser.add_argument('--num_workers', default=6, type=int, help='number of workers for the data loader')
        parser.add_argument('--pin_memory', default=0, type=int, help='whether to use pin_memory for the data loader')

        parser.add_argument('--train_ratio', default=.8, type=float, help='ratio for train/test split (when no k-fold)')
        parser.add_argument('--kfold', default=0, type=int,
                            help='If non zero, number of folds for k-fold training, and overwrites train_ratio argument')
        parser.add_argument('--validation', default=0, type=int,
                            help='If set to 1 each epoch will be tested on a validation set of same length as the test set,'
                                 ' and the best epoch will be used for the final test on a separate test set')

        # Weight saving
        parser.add_argument('--save_last', default=1, type=int,
                            help='If 1 (default), will only save the weights of the last testing epoch')
        parser.add_argument('--save_all', default=0, type=int,
                            help='If 1, will save the weights of all testing steps.')
        parser.add_argument('--save_best', default=0, type=int,
                            help='If 1, will only save the weights of the best epoch')
        parser.add_argument('--metric_best', default='IoU', type=str,
                            help='metric used to rank the epoch performances, chose between acc / loss / IoU(default)')

        # Optimization parameters
        parser.add_argument('--epochs', default=1000, type=int)
        parser.add_argument('--batch_size', default=128, type=int, help='Batch size')
        parser.add_argument('--lr', default=1e-3, type=float, help='Initial learning rate')
        # parser.add_argument('--lr_decay', default=1, type=float,   TODO
        #                     help='Multiplicative factor used on learning rate at lr_steps')
        # parser.add_argument('--lr_steps', default='',
        #                     help='List of epochs where the learning rate is decreased by `lr_decay`, separate with a +')

        parser.add_argument('--test_epoch', default=10, type=int, help='Test model every that many epochs')
        parser.add_argument('--display_step', default=100, type=int,
                            help='Display progress within one epoch every that many steps')

        parser.add_argument('--shuffle', default=True, help='Shuffle dataset')
        parser.add_argument('--grad_clip', default=0, type=float,
                            help='If nonzero, absolute value of the gradients will be clipped at this value')

        self.parser = parser

    def add_arguments(self, arg_dict):
        """
        Adds custom arguments to the argparse menu
        Args:
            arg_dict (dict): {arg_name:{'default':default_value,'type':arg_type}
        """
        for name, setting in arg_dict.items():
            self.parser.add_argument('--{}'.format(name), default=setting['default'], type=setting['type'])

    def parse_args(self):
        """Parses the arguments passed in the command line"""
        self.args = self.parser.parse_args()

    def print_args(self):
        """Displays the values of the arguments """
        print(self.args)

    def _check_args_consistency(self):
        """Performs several sanity checks on the arguments passed to the rack.
        Inconsistencies are automatically resolved, and a warning is prompted, but no exception is thrown."""
        if (self.args.save_best + self.args.save_all) == 1:
            self.args.save_last = 0

        if self.args.validation and self.args.test_epoch != 1:
            print('[WARNING] Validation requires testing at each epoch, setting test_epoch to 1')
            self.args.test_epoch = 1
            if (self.args.save_best + self.args.save_all) == 0:
                print('[WARNING] Validation requires save_all or save_best, setting save_best to 1')
                self.args.save_best = 1
                self.args.save_all = 0

        if (self.args.save_last + self.args.save_best) > 1:
            print('[WARNING] save_best and save_all are mutually exclusive setting save_all to 0')
            self.args.save_all = 0

        if self.args.kfold != 0:
            if self.args.kfold < 3 and self.args.validation:
                print('[WARNING] K-fold training with validation requires k > 2, setting k=3')
                self.args.kfold = 3

        if self.args.device == 'cuda' and not torch.cuda.is_available():
            print('[WARNING] No GPU found, setting device to gpu')
            self.args.device = 'cpu'

    ####### Methods for setting the specific elements of the training rack

    def _set_device(self):
        """Sets the device used by torch for computation"""
        self.device = torch.device(self.args.device)

    def add_model_configs(self, model_configs):
        """
        Add model configurations to be trained in the current script.
        Args:
            model_configs (list): list of trainRack.ModelConfig instances
        """
        self.model_configs.extend(model_configs)

    def set_dataset(self, dataset):
        """
        Attaches a custom dataset (inheriting from the torch.utils.data.Dataset class) to the training rack.
        The dataset should return items in the form (input,target).

        If data augmentation is used and should be only active during training, a list of two dataset instances can be
        passed. Both datasets should point to the same data (the train/val/test split is managed by the rack) but one
        should have the augmentation turned off.

        Args:
            dataset(torch.utils.data.Dataset, or list thereof): Dataset to be used for training
        """
        if isinstance(dataset, torch.utils.data.Dataset):
            self.dataset_train_transforms = dataset
            self.dataset_test_transforms = dataset
        elif isinstance(dataset, list):
            self.dataset_train_transforms, self.dataset_test_transforms = dataset

    ####### Methods for preparation

    def _prepare_output(self):
        """
        Creates the output directory (specified by the --res_dir parameter), with one sub-folder per model and
        writes the a summary of the configuration in each of them.
        """

        conf = self.to_dict()

        for mc in self.model_configs:

            res_dir = os.path.join(self.args.res_dir, mc.name)
            mc.res_dir = res_dir

            if not os.path.exists(res_dir):
                os.makedirs(res_dir)
            else:
                print('[WARNING] Output directory  already exists')

            if self.args.kfold != 0:
                for i in range(self.args.kfold):
                    os.makedirs(os.path.join(res_dir, 'FOLD_{}'.format(i + 1)), exist_ok=True)

            with open(os.path.join(res_dir, 'conf.json'), 'w') as fp:
                json.dump(conf[mc.name], fp, indent=4)

    def _get_loaders(self):
        """
        Prepares the sequence of train/(val)/test loaders that will be used. The sequence will be of length one if
        --k_fold is left to 0 (simple training).
        """
        if self.args.pin_memory == 1:
            pm = True
        else:
            pm = False

        print('[DATASET] Splitting dataset')

        indices = list(range(len(self.dataset_train_transforms)))

        if self.args.shuffle:
            np.random.seed(128)  # TODO random seed as an option
            np.random.shuffle(indices)

        # TRAIN / TEST SPLIT
        if self.args.kfold != 0:
            kf = model_selection.KFold(n_splits=self.args.kfold, random_state=1, shuffle=False)
            indices_seq = list(kf.split(list(range(len(indices)))))
            self.ntrain = len(indices_seq[0][0])
            self.ntest = len(indices_seq[0][1])
            print('[TRAINING CONFIGURATION] Preparing {}-fold cross validation'.format(self.args.kfold))

        else:
            self.ntrain = int(np.floor(self.args.train_ratio * len(self.dataset_train_transforms)))
            self.ntest = len(self.dataset_train_transforms) - self.ntrain
            indices_seq = [(list(range(self.ntrain)), list(range(self.ntrain, self.ntrain + self.ntest, 1)))]

        print('[DATASET] Train: {} samples, Test : {} samples'.format(self.ntrain, self.ntest))

        loader_seq = []
        record = []

        # TRAIN / VALIDATION SPLIT AND LOADER PREPARATION
        for train, test in indices_seq:
            train_indices = [indices[i] for i in train]
            test_indices = [indices[i] for i in test]

            if self.args.validation:
                validation_indices = train_indices[-self.ntest:]
                train_indices = train_indices[:-self.ntest]

                record.append((train_indices, validation_indices, test_indices))

                train_sampler = data.sampler.SubsetRandomSampler(train_indices)
                validation_sampler = data.sampler.SubsetRandomSampler(validation_indices)
                test_sampler = data.sampler.SubsetRandomSampler(test_indices)

                train_loader = data.DataLoader(self.dataset_train_transforms, batch_size=self.args.batch_size,
                                               sampler=train_sampler,
                                               num_workers=self.args.num_workers, pin_memory=pm)
                validation_loader = data.DataLoader(self.dataset_test_transforms, batch_size=self.args.batch_size,
                                                    sampler=validation_sampler,
                                                    num_workers=self.args.num_workers, pin_memory=pm)
                test_loader = data.DataLoader(self.dataset_test_transforms, batch_size=self.args.batch_size,
                                              sampler=test_sampler,
                                              num_workers=self.args.num_workers, pin_memory=pm)
                loader_seq.append((train_loader, validation_loader, test_loader))

            else:

                record.append((train_indices, test_indices))

                train_sampler = data.sampler.SubsetRandomSampler(train_indices)
                test_sampler = data.sampler.SubsetRandomSampler(test_indices)

                train_loader = data.DataLoader(self.dataset_train_transforms, batch_size=self.args.batch_size,
                                               sampler=train_sampler,
                                               num_workers=self.args.num_workers, pin_memory=pm)
                test_loader = data.DataLoader(self.dataset_test_transforms, batch_size=self.args.batch_size,
                                              sampler=test_sampler,
                                              num_workers=self.args.num_workers, pin_memory=pm)
                loader_seq.append((train_loader, test_loader))

        pkl.dump(record, open(os.path.join(self.args.res_dir, 'Dataset_split.pkl'), 'wb'))

        return loader_seq

    def _init_trainlogs(self):
        """Prepares the trainlog dictionaries."""
        self.stats = {}
        self.best_performance = {}

        for mc in self.model_configs:
            self.stats[mc.name] = {}
            self.best_performance[mc.name] = {'epoch': 0, 'IoU': 0, 'acc': 0, 'loss': sys.float_info.max}

    def _models_to_device(self):
        """Sends the models to the specified device."""
        for mc in self.model_configs:
            mc.model = mc.model.to(self.device)

    def _init_weights(self):
        """Initializes the weights of the rack's models."""
        for mc in self.model_configs:
            mc.model = mc.model.apply(weight_init)

    ####### Methods for execution

    def launch(self):  # TODO
        """
        Main method of the module: Prepares training (check arguments consistency, prepare output directories, split
        dataset and retrieve dataloaders) and performs training and testing according to the arguments specified in the
        argparse menu.
        """

        self._check_args_consistency()
        self._prepare_output()
        self._set_device()

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
                print('[TRAINING CONFIGURATION] Starting training for fold {}/{}'.format(i + 1, nfold))
                subdir = 'FOLD_{}'.format(i + 1)

            self.args.total_step = len(self.train_loader)

            self._init_weights()

            self._init_trainlogs()

            self._models_to_device()

            for self.current_epoch in range(1, self.args.epochs + 1):
                t0 = time.time()

                train_metrics = self._train_epoch()

                self._checkpoint_epoch(self.current_epoch, train_metrics, subdir=subdir)

                t1 = time.time()

                print('[PROGRESS] Epoch duration : {}'.format(t1 - t0))
                print('####################################################')

    def _train_epoch(self):  # TODO
        """
        Trains the model(s) on one epoch and displays the evolution of the training metrics.
        Returns a dictionary with the performance metrics over the whole epoch.
        """
        acc_meter = {}
        loss_meter = {}

        y_true = []
        y_pred = {m.name: [] for m in self.model_configs}

        for mc in self.model_configs:
            mc.model = mc.model.train()
            acc_meter[mc.name] = tnt.meter.ClassErrorMeter(accuracy=True)
            loss_meter[mc.name] = tnt.meter.AverageValueMeter()

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

            for mc in self.model_configs:

                outputs[mc.name] = mc.model(x)
                loss[mc.name] = mc.criterion(outputs[mc.name], y.long())
                prediction[mc.name] = outputs[mc.name].detach()

                acc_meter[mc.name].add(prediction[mc.name], y)
                loss_meter[mc.name].add(loss[mc.name].item())

                y_p = prediction[mc.name].argmax(dim=1).cpu().numpy()
                y_pred[mc.name].extend(list(y_p))

                mc.optimizer.zero_grad()
                loss[mc.name].backward()

                if self.args.grad_clip > 0:
                    for p in mc.model.parameters():
                        p.grad.data.clamp_(-self.args.grad_clip, self.args.grad_clip)

                mc.optimizer.step()

                if (i + 1) % self.args.display_step == 0:
                    tb = time.time()
                    elapsed = tb - ta
                    print('[{:20.20}] Step [{}/{}], Loss: {:.4f}, Acc : {:.2f}, Duration:{:.4f}'
                          .format(mc.name, i + 1, self.args.total_step, loss_meter[mc.name].value()[0],
                                  acc_meter[mc.name].value()[0],
                                  elapsed))
                    ta = tb

        metrics = {}
        for mc in self.model_configs:
            miou = mIou(y_true, y_pred[mc.name], self.args.num_classes)
            metrics[mc.name] = {'loss': loss_meter[mc.name].value()[0],
                                'accuracy': acc_meter[mc.name].value()[0],
                                'IoU': miou}

        return metrics

    def _checkpoint_epoch(self, epoch, metrics, subdir=''):
        """
        Writes the training metrics - and testing metrics, if on a test epoch - in the output trainlog file, and saves
        the model(s) weights according to the strategy specified in the argparse menu (save best, last or all).

        Args:
            epoch (int): epoch number
            metrics (dict): training metrics to be written
            subdir (str): Optional, name of the target sub directory (used for k-fold)
        """

        if epoch % self.args.test_epoch == 0 or epoch == self.args.epochs:  # Test epoch or last epoch

            if epoch == self.args.epochs:
                test_metrics, (y_true, y_pred) = self._test(return_y=True)
            else:
                test_metrics = self._test(return_y=False)

            for mc in self.model_configs: # Write trainlog and model weights

                self.stats[mc.name][epoch] = {**metrics[mc.name], **test_metrics[mc.name]}  # TODO
                print('[PROGRESS - {}] Writing checkpoint of epoch {}\{} . . .'.format(mc.name, epoch,
                                                                                       self.args.epochs))

                with open(os.path.join(mc.res_dir, subdir, 'trainlog.json'), 'w') as outfile:
                    json.dump(self.stats[mc.name], outfile, indent=4)

                if self.args.save_best == 1:
                    if self.best_performance[mc.name]['epoch'] == self.current_epoch:
                        file_name = 'model.pth.tar'
                        torch.save({'epoch': epoch, 'state_dict': mc.model.state_dict(),
                                    'optimizer': mc.optimizer.state_dict()},
                                   os.path.join(mc.res_dir, subdir, file_name))
                else:
                    if self.args.save_all == 1:
                        file_name = 'model_epoch{}.pth.tar'.format(epoch)
                    else:
                        file_name = 'model.pth.tar'
                    torch.save({'epoch': epoch, 'state_dict': mc.model.state_dict(),
                                'optimizer': mc.optimizer.state_dict()},
                               os.path.join(mc.res_dir, subdir, file_name))

            if epoch == self.args.epochs:  #Final epoch

                if self.args.validation:
                    y_true, y_pred = self._get_best_predictions(subdir=subdir)

                for mc in self.model_configs: # Final performance on the test set
                    per_class, conf_m = self._final_performance(y_true, y_pred[mc.name])
                    with open(os.path.join(mc.res_dir, subdir, 'per_class_metrics.json'), 'w') as outfile:
                        json.dump(per_class, outfile, indent=4)
                    pkl.dump(conf_m, open(os.path.join(mc.res_dir, subdir, 'confusion_matrix.pkl'), 'wb'))

        else:  #Regular epoch without testing
            for mc in self.model_configs:
                self.stats[mc.name][epoch] = metrics[mc.name]
                print(
                    '[PROGRESS - {}] Writing checkpoint of epoch {}\{} . . .'.format(mc.name, epoch, self.args.epochs))

                with open(os.path.join(mc.res_dir, subdir, 'trainlog.json'), 'w') as outfile:
                    json.dump(self.stats[mc.name], outfile, indent=4)

    def _test(self, return_y=False):  # TODO
        """
        Tests the model on the test or validation set and returns the performance metrics as a dictionary

        Args:
            return_y (bool): If True the method will also return the predicted and true labels on the test set.
        """
        # TODO generalise method and just give a list of metrics as rack parameter

        print('Testing models . . .')


        # Initialize meters

        acc_meter = {}
        loss_meter = {}

        y_true = []
        y_pred = {m.name: [] for m in self.model_configs}

        for mc in self.model_configs:
            mc.model = mc.model.eval()

            acc_meter[mc.name] = tnt.meter.ClassErrorMeter(accuracy=True)
            loss_meter[mc.name] = tnt.meter.AverageValueMeter()

        loader = self.validation_loader if self.args.validation else self.test_loader

        # Inference

        for (x, y) in loader:

            y_true.extend(list(map(int, y)))

            x = x.to(self.device)
            y = y.to(self.device)

            prediction = {}
            loss = {}
            for mc in self.model_configs:
                with torch.no_grad():
                    prediction[mc.name] = mc.model(x)
                    loss[mc.name] = mc.criterion(prediction[mc.name], y)

                acc_meter[mc.name].add(prediction[mc.name], y)
                loss_meter[mc.name].add(loss[mc.name].item())

                y_p = prediction[mc.name].argmax(dim=1).cpu().numpy()
                y_pred[mc.name].extend(list(y_p))

        test_metrics = {}

        for mc in self.model_configs: #Display, and archive best results in case of validation
            acc = acc_meter[mc.name].value()[0]
            loss = loss_meter[mc.name].value()[0]
            miou = mIou(y_true, y_pred[mc.name], self.args.num_classes)

            test_metrics[mc.name] = {'test_accuracy': acc, 'test_loss': loss, 'test_IoU': miou}

            mode = 'Test' if self.args.validation == 0 else 'Val'
            print('[{}] {} Loss: {:.4f}, Acc : {:.2f}, IoU {:.4f}'.format(mc.name, mode, loss, acc, miou))

            if self.args.validation:
                metric = [m for m in test_metrics[mc.name].keys() if self.args.metric_best in m][0]

                if self.args.metric_best == 'loss':
                    if test_metrics[mc.name][metric] < self.best_performance[mc.name][self.args.metric_best]:
                        print('[PERFORMANCE - {}] BEST EPOCH !'.format(mc.name))
                        self.best_performance[mc.name]['IoU'] = test_metrics[mc.name]['test_IoU']
                        self.best_performance[mc.name]['acc'] = test_metrics[mc.name]['test_accuracy']
                        self.best_performance[mc.name]['loss'] = test_metrics[mc.name]['test_loss']

                        self.best_performance[mc.name]['epoch'] = self.current_epoch
                else:
                    if test_metrics[mc.name][metric] > self.best_performance[mc.name][self.args.metric_best]:
                        print('[PERFORMANCE - {}] BEST EPOCH !'.format(mc.name))
                        self.best_performance[mc.name]['IoU'] = test_metrics[mc.name]['test_IoU']
                        self.best_performance[mc.name]['acc'] = test_metrics[mc.name]['test_accuracy']
                        self.best_performance[mc.name]['loss'] = test_metrics[mc.name]['test_loss']

                        self.best_performance[mc.name]['epoch'] = self.current_epoch

                test_metrics[mc.name] = {'val_accuracy': acc, 'val_loss': loss, 'val_IoU': miou}

        if return_y:
            return test_metrics, (y_true, y_pred)
        else:
            return test_metrics

    def _final_performance(self, y_true, y_pred):
        """
        Computes the final performance(s) of the model(s) on the test set. If trained with validation, the weights of
        the epoch achieving the best results are used. Otherwise, the weights of the last epoch are kept for testing.
        """
        per_class = per_class_performance(y_true, y_pred, self.args.num_classes)
        conf_m = conf_mat(y_true, y_pred, self.args.num_classes)

        return per_class, conf_m

    def _get_best_predictions(self, subdir=''):
        """
        Gets the prediction of the model on the test set, using the weights that achieved the best performance on the
        validation set.
        """
        y_true = []
        y_pred = {m.name: [] for m in self.model_configs}

        for mc in self.model_configs:
            best_epoch = self.best_performance[mc.name]['epoch']

            if self.args.save_all == 1:
                file_name = 'model_epoch{}.pth.tar'.format(best_epoch)
            else:
                file_name = 'model.pth.tar'

            checkpoint = torch.load(os.path.join(mc.res_dir, subdir, file_name.format(
                best_epoch)))
            mc.model.load_state_dict(checkpoint['state_dict'])
            mc.model.eval()

        for (x, y) in self.test_loader:

            y_true.extend(list(map(int, y)))
            x = x.to(self.device)

            for mc in self.model_configs:
                with torch.no_grad():
                    prediction = mc.model(x)
                    y_pred[mc.name].extend(prediction.argmax(dim=1).cpu().numpy())
        return y_true, y_pred


class ModelConfig:
    """
    Individual model configuration. It is defined by the model, the loss and the optimizer to be used.
    Attributes:
        name (str): Name of the configuration.
        model (torch.nn.Module): Instance of model to be trained.
        criterion (torch.nn loss): Instance of loss to use for training.
        optimizer (torch.optim otimizer): Instance of optmizer to use for training.
    """
    def __init__(self, name, model=None, criterion=None, optimizer=None):
        self.name = name
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.res_dir = None

    def set_model(self, model):
        self.model = model

    def set_criterion(self, criterion):
        self.criterion = criterion

    def set_optimizer(self, optimizer):
        self.optimizer = optimizer
