import torch
from torch.utils import data
from torch.optim.lr_scheduler import MultiStepLR
import torchnet as tnt
import numpy as np
from sklearn import model_selection

from izitorch.metrics import mIou
from izitorch.utils import weight_init

import argparse
import time
import json
import os


#TODO update docstring


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

            output[model_name] = d
        return output

    ####### Methods for the options menu
    def set_basic_menu(self):
        """
        Sets a basic argparse menu with the common arguments
        """
        parser = argparse.ArgumentParser()

        parser.add_argument('--dataset', default='/home/vsfg/data')
        parser.add_argument('--res_dir', default='results', help='folder for saving the trained model')
        parser.add_argument('--resume', default='')

        parser.add_argument('--batch_size', default=128, type=int, help='Batch size')
        parser.add_argument('--shuffle', default=True, help='Shuffle dataset')
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

    ####### Methods for setting the specific elements of the training rack

    def set_device(self, device_name=None):
        """Sets the device used by torch for computation, will prioritize GPU by default but can be set manually"""
        try:
            self.device = torch.device(device_name)
        except TypeError:
            self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    def set_models(self, model_dict):
        """
        Attaches a model instance to the training rack
        Args:
            model(dict): dictionary of instances of torch.nn.Module (items in the form name:instance)
        """
        self.models = model_dict

    def set_optimizers(self, optimizer_classes):
        """
        Attaches an optimizer instance to the training rack
        Args:
            optimizer_classes (dict): list of classes (NOT INSTANCE) of the optimizer to be used for each model
        """
        self.optimizers = {}

        for model_name, opt in optimizer_classes.items():
            self.optimizers[model_name] = opt(self.models[model_name].parameters(), lr=self.args.lr)

        # if self.args.lr_decay != 1:   #TODO add lr decay support
        #     print('[TRAINING CONFIGURATION] Preparing MutliStepLR')
        #     steps = list(map(int, self.args.lr_steps.split('+')))
        #     self.scheduler = MultiStepLR(self.optimizer, milestones=steps, gamma=self.args.lr_decay)

    def set_criterions(self, criterions):
        """
        Attaches a criterion instance to the training rack
        Args:
            criterion (dict): dict of instances of a torch.nn criterion, one for each model
        """
        self.criterions = criterions

    def add_model_configs(self, model_configs):
        """
        Add model configurations to be trainined in the current script
        Args:
            model_configs (dict): model configuration with entries in the form:
                        {
                            model_name:{
                                        'model':torch.nn instance,
                                        'optimizer': torch optimizer instance,
                                        'criterion' : torch.nn criterion instance
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

    def prepare_output(self):
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

    def get_loaders(self):
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
            np.random.seed(1)  # TODO random seed as an option
            np.random.shuffle(indices)

        if self.args.kfold != 0:
            kf = model_selection.KFold(n_splits=self.args.kfold, random_state=1, shuffle=False)
            indices_seq = list(kf.split(list(range(len(indices)))))
            print(
                '[DATASET] Train: {} samples, Test : {} samples'.format(len(indices_seq[0][0]), len(indices_seq[0][1])))
            print('[TRAINING CONFIGURATION] Preparing {}-fold cross validation'.format(self.args.kfold))

        else:
            ntrain = int(np.floor(self.args.train_ratio * len(self.train_dataset)))
            ntest = len(self.train_dataset) - ntrain
            print('[DATASET] Train: {} samples, Test : {} samples'.format(ntrain, ntest))
            indices_seq = [(list(range(ntrain)), list(range(ntrain, ntrain + ntest, 1)))]

        loader_seq = []

        for train, test in indices_seq:
            train_indices = [indices[i] for i in train]
            test_indices = [indices[i] for i in test]

            train_sampler = data.sampler.SubsetRandomSampler(train_indices)
            test_sampler = data.sampler.SubsetRandomSampler(test_indices)

            train_loader = data.DataLoader(self.train_dataset, batch_size=self.args.batch_size,
                                           sampler=train_sampler,
                                           num_workers=self.args.num_workers, pin_memory=pm)
            test_loader = data.DataLoader(self.test_dataset, batch_size=self.args.batch_size,
                                          sampler=test_sampler,
                                          num_workers=self.args.num_workers, pin_memory=pm)
            loader_seq.append((train_loader, test_loader))

        return loader_seq

    def initialise_weights(self):
        for conf in self.model_configs.values():
            conf['model'].apply(weight_init)

    ####### Methods for execution

    def launch(self):  # TODO
        """
        MAIN METHOD: does the necessary preparations and launches the training loop for the specified number of epochs
        the model is applied to the test dataset every args.test_step epochs
        """

        self.prepare_output()

        loader_seq = self.get_loaders()
        nfold = len(loader_seq)

        for i, (self.train_loader, self.test_loader) in enumerate(loader_seq):

            if nfold == 1:
                print('[TRAINING CONFIGURATION] Starting single training ')
                subdir = ''
            else:
                print('[TRAINING CONFIGURATION] Training with {}-fold cross validation'.format(nfold))
                subdir = 'FOLD_{}'.format(i + 1)

            self.args.total_step = len(self.train_loader)

            self.initialise_weights()

            self.init_trainlogs()

            self.models_to_device()

            print('[PROGRESS] FOLD #{}'.format(i + 1))
            for epoch in range(self.args.epochs):
                t0 = time.time()

                train_metrics = self.train_epoch()
                self.checkpoint_epoch(epoch, train_metrics, subdir=subdir)

                t1 = time.time()

                print('[PROGRESS] Epoch duration : {}'.format(t1 - t0))

    def init_trainlogs(self):
        self.stats = {}
        for model_name, conf in self.model_configs.items():
            self.stats[model_name] = {}

    def models_to_device(self):
        for model_name, conf in self.model_configs.items():
            conf['model'] = conf['model'].to(self.device)

    def train_epoch(self):  # TODO
        """
        Trains the model on one epoch and displays the evolution of the training metrics.
        Returns a dictionary with the performance metrics over the whole epoch.
        """
        acc_meter = {}
        loss_meter = {}

        for model_name, conf in self.model_configs.items():
            conf['model'].train()
            acc_meter[model_name] = tnt.meter.ClassErrorMeter(accuracy=True)
            loss_meter[model_name] = tnt.meter.AverageValueMeter()

        ta = time.time()

        for i, (x, y) in enumerate(self.train_loader):

            try:
                x = x.to(self.device)
            except AttributeError:  # dirty fix for extra data
                for j, input in enumerate(x):
                    x[j] = input.to(self.device)

            y = y.to(self.device)

            for model_name, conf in self.model_configs.items():
                outputs = conf['model'](x)
                loss = conf['criterion'](outputs, y.long())

                acc_meter[model_name].add(outputs.detach(), y)
                loss_meter[model_name].add(loss.item())

                conf['optimizer'].zero_grad()
                loss.backward()
                conf['optimizer'].step()

                if (i + 1) % 100 == 0:
                    tb = time.time()
                    elapsed = tb - ta
                    print('[PROGRESS - MODEL {}] Step [{}/{}], Loss: {:.4f}, Accuracy : {:.3f}, Elapsed time:{:.2f}'
                          .format(model_name, i + 1, self.args.total_step, loss_meter[model_name].value()[0],
                                  acc_meter[model_name].value()[0],
                                  elapsed))
                    ta = tb

        metrics = {}
        for model_name in self.model_configs:
            metrics[model_name] = {'loss': loss_meter[model_name].value()[0],
                                   'accuracy': acc_meter[model_name].value()[0]}
        return metrics

    def checkpoint_epoch(self, epoch, metrics, subdir=''):  # TODO
        """
        Computes accuracy and loss of the epoch and writes it in the trainlog, along with the model weights.
        If on a test epoch the test metrics will also be computed and writen in the trainlog
        Args:
            epoch (int): number of the epoch
            metrics (dict): training metrics to be written
            subdir (str): Optional, name of the target sub directory (used for k-fold)
        """
        if epoch % self.args.test_step == 0 or epoch == self.args.epochs:
            test_metrics = self.test()
            for model_name in self.model_configs:
                self.stats[model_name][epoch + 1] = {**metrics[model_name], **test_metrics[model_name]} #TODO
        else:
            for model_name, conf in self.model_configs.items():
                self.stats[model_name][epoch + 1] = metrics[model_name]

                print('[PROGRESS - MODEL {}] Writing checkpoint of epoch {}\{} . . .'.format(model_name, epoch + 1,
                                                                                             self.args.epochs))

                with open(os.path.join(conf['res_dir'], subdir, 'trainlog.json'), 'w') as outfile:
                    json.dump(self.stats[model_name], outfile)
                torch.save(
                    {'epoch': epoch + 1, 'state_dict': conf['model'].state_dict(),
                     'optimizer': conf['optimizer'].state_dict()},
                    os.path.join(conf['res_dir'], subdir, 'model.pth.tar'.format(epoch)))


    def test(self):  # TODO
        """
        Tests the model on the test set and returns the performance metrics as a dictionnary
        """
        # TODO generalise method and just give a list of metrics as rack parameter
        print('Testing models . . .')

        acc_meter = {}
        loss_meter = {}
        iou_meter = {}

        for model_name, conf in self.model_configs.items():
            conf['model'].eval()
            acc_meter[model_name] = tnt.meter.ClassErrorMeter(accuracy=True)
            loss_meter[model_name] = tnt.meter.AverageValueMeter()
            iou_meter[model_name] = tnt.meter.AverageValueMeter()

        for (x, y) in self.test_loader:
            x = x.to(self.device)
            y = y.to(self.device)

            for model_name, conf in self.model_configs.items():

                with torch.no_grad():
                    prediction = conf['model'](x)
                    loss = conf['criterion'](prediction, y)

                acc_meter[model_name].add(prediction, y)
                loss_meter[model_name].add(loss.item())

                iou = mIou(y.cpu().numpy(), (prediction.argmax(dim=1).cpu().numpy()), n_classes=self.args.num_classes)

                iou_meter[model_name].add(iou)

        test_metrics = {}
        for model_name in self.model_configs:

            acc = acc_meter[model_name].value()[0]
            loss = loss_meter[model_name].value()[0]
            miou = iou_meter[model_name].value()[0]

            test_metrics[model_name] = {'test_accuracy': acc, 'test_loss': loss, 'test_IoU': miou}

            print('[PERFORMANCE - MODEL {}] Test accuracy : {:.3f}'.format(model_name,acc))
            print('[PERFORMANCE - MODEL {}] Test loss : {:.4f}'.format(model_name,loss))
            print('[PERFORMANCE - MODEL {}] Test IoU : {:.4f}'.format(model_name,miou))

        return test_metrics


