"""
Example script using the trainRack, used for testing.
(No animals were harmed in the process)
"""

import torch.nn as nn
import torch
from torch.utils import data

from izitorch.trainRack import Rack, ModelConfig


class Model(nn.Module):
    def __init__(self, hs, ins, nc):
        super(Model, self).__init__()
        self.rec = nn.GRU(input_size=ins, hidden_size=hs, batch_first=True)
        self.cla = nn.Linear(hs, nc)

    def forward(self, input, hx=None):
        out, hn = self.rec(input)
        return nn.Softmax(dim=-1)(self.cla(hn[-1, :, :]))


class RandomDataset(data.Dataset):
    def __init__(self, nsamp=1000, seqlen=5, nfeat=10, nclass=2):
        super(RandomDataset, self).__init__()
        self.nsamp = nsamp
        self.nfeat = nfeat
        self.data = torch.randn(nsamp, seqlen, nfeat)
        self.target = torch.randint(nclass, (nsamp,))

    def __getitem__(self, item):
        return self.data[item, :, :], self.target[item].long()

    def __len__(self):
        return self.nsamp


arg_dict = {'hidden_size': {'default': 32, 'type': int}}

if __name__ == '__main__':
    rack = Rack()

    rack.add_arguments(arg_dict)
    rack.parse_args()

    m1 = Model(rack.args.hidden_size, 10, 2)
    m2 = Model(rack.args.hidden_size, 10, 2)

    confs = [
        ModelConfig('model1', model=m1, criterion=nn.CrossEntropyLoss(),
                    optimizer=torch.optim.Adam(m1.parameters())),
        ModelConfig('model2', model=m1, criterion=nn.CrossEntropyLoss(),
                    optimizer=torch.optim.Adam(m2.parameters()))
    ]

    rack.add_model_configs(confs)

    rack.set_dataset(RandomDataset())

    rack.launch()
