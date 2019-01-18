import torch.nn as nn
import torch
from torch.utils import data

from izitorch import trainRack


class ExModel(nn.Module):
    def __init__(self, hs, ins, nc):
        super(ExModel, self).__init__()
        self.rec = nn.GRU(input_size=ins, hidden_size=hs, batch_first=True)
        self.cla = nn.Linear(hs, nc)

    def forward(self, input, hx=None):
        out, hn = self.rec(input)
        return nn.Softmax(dim=-1)(self.cla(hn[-1, :, :]))




class RandDataset(data.Dataset):
    def __init__(self, nsamp=10000, seqlen=5, nfeat=10, nclass=2):
        super(RandDataset, self).__init__()
        self.nsamp = nsamp
        self.nfeat = nfeat
        self.data = torch.randn(nsamp, seqlen, nfeat)
        self.target = torch.randint(nclass, (nsamp,))

    def __getitem__(self, item):
        return self.data[item, :, :], self.target[item].long()

    def __len__(self):
        return self.nsamp


arg_dict = {'hidden_size': {'default': 64, 'type': int}}



if __name__ == '__main__':
    rack = trainRack.Rack()

    rack.add_arguments(arg_dict)
    rack.parse_args()

    m1 = ExModel(rack.args.hidden_size, 10, 2)
    conf = {
        'model1': {
            'model': m1,
            'criterion': nn.CrossEntropyLoss(),
            'optimizer': torch.optim.Adam(m1.parameters())
        }
    }

    rack.add_model_configs(conf)

    rack.set_dataset(RandDataset())

    rack.launch()
