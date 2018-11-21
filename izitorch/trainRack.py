import torch
from torch.utils import data
import torchnet as tnt
import numpy as np

from izitorch.metrics import mIou

import argparse
import time
import json
import os


class Rack:

    def __init__(self):
        torch.manual_seed(1)

        self.set_basic_menu()
        self.set_device()
        self.dataset = None

    def to_dict(self):
        d = vars(self.args).copy()
        d['model'] = str(self.model)
        d['criterion'] = str(self.criterion)
        d['optimizer'] = str(self.optimizer)
        d['device'] = str(self.device)
        return d

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

        parser.add_argument('--epochs', default=1000, type=int)
        parser.add_argument('--lr', default=1e-3, type=float, help='Initial learning rate')
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

    def set_model(self, model):
        """
        Attaches a model instance to the training rack
        Args:
            model: instance of torch.nn.Module
        """
        self.model = model

    def set_optimizer(self, optimizer_class):
        """
        Attaches an optimizer instance to the training rack
        Args:
            optimizer_class: class (NOT INSTANCE) of the optimizer to be used
        """
        self.optimizer = optimizer_class(self.model.parameters(), lr=self.args.lr)

    def set_criterion(self, criterion):
        """
        Attaches a criterion instance to the training rack
        Args:
            criterion: instance of a torch.nn criterion
        """
        self.criterion = criterion

    def set_dataset(self, dataset):
        """
        Attaches a dataset to the training rack. If only one dataset is provided, it will be split in train and test,
        but a list of two datasets can also be provided (allows to have different image transformation on train and test).
        For generality's sake the dataset should return items in the form (input,target).
        If the network uses more exotic input, this needs to be dealt with in the forward method of its definition
        Args:
            dataset: instance of torch.utils.dataset or list [train_dataset, test_dataset]
        """
        if isinstance(dataset,torch.utils.data.Dataset):
            self.dataset = dataset
        elif isinstance(dataset,list):
            self.train_dataset , self.test_dataset = dataset
        else:
            raise ValueError

    ####### Methods for preparation

    def prepare_output(self):
        """
        Creates output directory and writes the configuration file in it.
        Then initializes the dictionnary that keep tracks of the metrics.
        """
        if not os.path.exists(self.args.res_dir):
            os.makedirs(self.args.res_dir)
        else:
            print('WARNING: Output directory  already exists')

        with open(os.path.join(self.args.res_dir, 'conf.json'), 'w') as fp:
            json.dump(self.to_dict(), fp, indent=4)

        self.stats = {}

    def get_loaders(self):
        """
        Splits the dataset in train and test if only one was provided,
         and returns the two corresponding torch.utils.DataLoader instances
        """
        print('Splitting dataset')

        ntrain = int(np.floor(self.args.train_ratio * len(self.dataset)))
        ntest = len(self.dataset) - ntrain
        print('Train: {} samples, Test : {} samples'.format(len(self.train_dataset), len(self.test_dataset)))

        if self.dataset is not None:

            self.train_dataset, self.test_dataset = data.random_split(self.dataset, [ntrain, ntest])


            train_loader = data.DataLoader(self.train_dataset, batch_size=self.args.batch_size, shuffle=self.args.shuffle,
                                           num_workers=self.args.num_workers)
            test_loader = data.DataLoader(self.test_dataset, batch_size=self.args.batch_size, shuffle=self.args.shuffle,
                                          num_workers=self.args.num_workers)

        else:
            indices = range(len(self.train_dataset))
            if self.args.shuffle:
                np.random.shuffle(indices)

            train_sampler = data.sampler.SubsetRandomSampler(indices[:ntrain])
            test_sampler = data.sampler.SubsetRandomSampler(indices[ntrain:])


            train_loader = data.DataLoader(self.train_dataset, batch_size=self.args.batch_size, sampler=train_sampler,
                                           num_workers=self.args.num_workers)
            test_loader = data.DataLoader(self.test_dataset, batch_size=self.args.batch_size, sampler=test_sampler,
                                          num_workers=self.args.num_workers)
        return train_loader, test_loader

    ####### Methods for execution

    def launch(self):
        """
        MAIN METHOD: does the necessary preparations and launches the training loop for the specified number of epochs
        the model is applied to the test dataset every args.test_step epochs
        """

        self.prepare_output()

        self.train_loader, self.test_loader = self.get_loaders()

        self.args.total_step = len(self.train_loader)

        self.model = self.model.to(self.device)

        for epoch in range(self.args.epochs):
            train_metrics = self.train_epoch()
            self.checkpoint_epoch(epoch, train_metrics)

    def checkpoint_epoch(self, epoch, metrics):
        """
        Computes accuracy and loss of the epoch and writes it in the trainlog, along with the model weights.
        If on a test epoch the test metrics will also be computed and writen in the trainlog
        Args:
            epoch (int): number of the epoch
            metrics (dict): training metrics to be written
        """
        if epoch % self.args.test_step == 0:
            test_metrics = self.test()
            self.stats[epoch + 1] = {**metrics, **test_metrics}

        else:
            self.stats[epoch + 1] = metrics

        print('Writing checkpoint . . .')
        with open(os.path.join(self.args.res_dir, 'trainlog.json'), 'w') as outfile:
            json.dump(self.stats, outfile)
        torch.save(
            {'epoch': epoch + 1, 'state_dict': self.model.state_dict(), 'optimizer': self.optimizer.state_dict()},
            os.path.join(self.args.res_dir, 'model.pth.tar'.format(epoch)))

    def train_epoch(self):
        """
        Trains the model on one epoch and displays the evolution of the training metrics.
        Returns a dictionary with the performance metrics over the whole epoch.
        """
        self.model.train()

        acc_meter = tnt.meter.ClassErrorMeter(accuracy=True)
        loss_meter = tnt.meter.AverageValueMeter()

        ta = time.time()
        for i, (x, y) in enumerate(self.train_loader):

            x = x.to(self.device)
            y = y.to(self.device)

            outputs = self.model(x)
            loss = self.criterion(outputs, y.long())

            acc_meter.add(outputs.detach(), y)
            loss_meter.add(loss.item())

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if (i + 1) % 100 == 0:
                tb = time.time()
                elapsed = tb - ta
                print('Step [{}/{}], Loss: {:.4f}, Accuracy : {:.3f}, Elapsed time:{:.2f}'
                      .format(i + 1, self.args.total_step, loss_meter.value()[0], acc_meter.value()[0],
                              elapsed))
                ta = tb

        return {'loss': loss_meter.value()[0], 'accuracy': acc_meter.value()[0]}

    def test(self):
        """
        Tests the model on the test set and returns the performance metrics as a dictionnary
        """
        # TODO generalise method and just give a list of metrics as rack parameter
        print('Testing model . . .')
        acc_meter = tnt.meter.ClassErrorMeter(accuracy=True)
        loss_meter = tnt.meter.AverageValueMeter()
        iou_meter = tnt.meter.AverageValueMeter()

        self.model.eval()
        for (x, y) in self.test_loader:
            x = x.to(self.device)
            y = y.to(self.device)

            with torch.no_grad():
                prediction = self.model(x)
                loss = self.criterion(prediction, y)

            acc_meter.add(prediction, y)
            loss_meter.add(loss.item())

            iou = mIou(y.cpu().numpy(), (prediction.argmax(dim=1).cpu().numpy()), n_classes=self.args.num_classes)

            iou_meter.add(iou)

        acc = acc_meter.value()[0]
        loss = loss_meter.value()[0]
        miou = iou_meter.value()[0]

        test_metrics = {'test_accuracy': acc, 'test_loss': loss, 'test_IoU': miou}

        print('Test accuracy : {:.3f}'.format(acc))
        print('Test loss : {:.4f}'.format(loss))
        print('Test IoU : {:.4f}'.format(miou))

        return test_metrics
