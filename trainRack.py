import torch #TODO minimal import
from torch.utils import data

import argparse

import json
import os


class Rack:

    def __init__(self):
        self.set_basic_menu()
        self.set_device()

    def to_dict(self):
        d = vars(self.args).copy()
        d['model'] = str(self.model)
        d['criterion'] = str(self.criterion)
        d['optimizer'] = str(self.optimizer)
        d['device'] = str(self.device)
        return d

    ####### Methods for the options menu
    def set_basic_menu(self):
        parser = argparse.ArgumentParser()

        parser.add_argument('--dataset', default='/home/vsfg/data')
        parser.add_argument('--res_dir', default='results', help='folder for saving the trained model')
        parser.add_argument('--resume', default='')

        parser.add_argument('--num_classes', default=44, type=int)

        parser.add_argument('--batch_size', default=128, type=int, help='Batch size')
        parser.add_argument('--shuffle', default=True, help='Shuffle dataset')
        parser.add_argument('--num_workers', default=6, type=int, help='number of workers for data loader')
        parser.add_argument('--train_ratio', default=.8, type=float, help='ratio for train/test split')

        parser.add_argument('--epochs', default=1000, type=int)
        parser.add_argument('--lr', default=1e-3, type=float, help='Initial learning rate')
        parser.add_argument('--test_step', default=10, type=int, help='Test model every that many steps')

        self.parser = parser

    def add_arguments(self, arg_dict):
        for name, setting in arg_dict.items():
            self.parser.add_argument('--{}'.format(name), default=setting['default'], type=setting['type'])

    def parse_args(self):
        self.args = self.parser.parse_args()

    def print_args(self):
        print(self.args)

    ####### Methods for setting the specific elements of the training rack

    def set_device(self, device_name=None):
        try:
            self.device(torch.device(device_name))
        except TypeError:
            self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    def set_model(self, model):
        self.model = model

    def set_optimizer(self, optimizer_class):
        self.optimizer = optimizer_class(self.model.parameters())

    def set_criterion(self, criterion):
        self.criterion = criterion

    def set_dataset(self, dataset):
        self.dataset = dataset

    ####### Methods for execution

    def prepare_output(self):
        if not os.path.exists(self.args.res_dir):
            os.makedirs(self.args.res_dir)
        else:
            print('WARNING: Output directory  already exists')

        with open(os.path.join(self.args.res_dir, 'conf.json'), 'w') as fp:
            json.dump(self.to_dict(), fp, indent=4)

    def get_loaders(self):

        print('Splitting dataset')
        ntrain = int(np.floor(self.args.train_ratio * len(self.dataset)))
        ntest = len(self.dataset) - ntrain

        dtrain, dtest = data.random_split(self.dataset, [ntrain, ntest])
        print('Train: {} samples, Test : {} samples'.format(ntrain, ntest))

        train_loader = data.DataLoader(dtrain, batch_size=self.args.batch_size, shuffle=self.args.shuffle,
                                       num_workers=self.args.num_workers)
        test_loader = data.DataLoader(dtest, batch_size=self.args.batch_size, shuffle=self.args.shuffle,
                                      num_workers=self.args.num_workers)
        return train_loader, test_loader

    def launch(self):
        self.prepare_output()

        self.train_loader, self.test_loader = self.get_loaders()

        self.train()

    def train(self):
        pass

    def test(self):
        pass
