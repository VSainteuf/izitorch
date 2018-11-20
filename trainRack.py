import torch  # TODO minimal import
from torch.utils import data
import torchnet as tnt
import numpy as np

from metrics import mIou

import argparse
import time
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
            self.device = torch.device(device_name)
        except TypeError:
            self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    def set_model(self, model):
        self.model = model

    def set_optimizer(self, optimizer_class):
        self.optimizer = optimizer_class(self.model.parameters(), lr=self.args.lr)

    def set_criterion(self, criterion):
        self.criterion = criterion

    def set_dataset(self, dataset):
        self.dataset = dataset

    ####### Methods for preparation

    def prepare_output(self):
        if not os.path.exists(self.args.res_dir):
            os.makedirs(self.args.res_dir)
        else:
            print('WARNING: Output directory  already exists')

        with open(os.path.join(self.args.res_dir, 'conf.json'), 'w') as fp:
            json.dump(self.to_dict(), fp, indent=4)

        self.stats = {}

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

    ####### Methods for execution

    def launch(self):
        torch.manual_seed(1)

        self.prepare_output()

        self.train_loader, self.test_loader = self.get_loaders()

        self.args.total_step = len(self.train_loader)

        self.model = self.model.to(self.device)

        for epoch in range(self.args.epochs):

            train_metrics = self.train_epoch()
            self.monitor_epoch(epoch, train_metrics)

    def monitor_epoch(self, epoch, metrics):
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

        #TODO generalise method and just give a list of metrics as rack parameter
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

        test_metrics = {'test_accuracy':acc,'test_loss':loss,'test_IoU':miou}

        print('Test accuracy : {:.3f}'.format(acc))
        print('Test loss : {:.4f}'.format(loss))
        print('Test IoU : {:.4f}'.format(miou))

        return test_metrics
