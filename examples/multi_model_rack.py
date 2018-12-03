import torch.nn as nn
import torch
from torch.utils import data

from izitorch import trainRack


class ExModel(nn.Module):
    def __init__(self,hs,ins,nc):
        super(ExModel,self).__init__()
        self.rec = nn.GRU(input_size=ins,hidden_size=hs,batch_first=True)
        self.cla = nn.Linear(hs,nc)

    def forward(self, input, hx=None):
        out, hn = self.rec(input)
        return nn.Softmax(dim=-1)(self.cla(hn[-1,:,:]))


class RandDataset(data.Dataset):
    def __init__(self, nsamp=100000, seqlen=5, nfeat=10, nclass=2):
        super(RandDataset, self).__init__()
        self.nsamp = nsamp
        self.nfeat = nfeat
        self.data = torch.randn(nsamp, seqlen, nfeat)
        self.target = torch.randint(nclass,(nsamp,))

    def __getitem__(self, item):
        return self.data[item, :, :], self.target[item].long()

    def __len__(self):
        return self.nsamp


arg_dict = {'num_classes': {'default': 2, 'type': int}}


m1 = ExModel(50, 10, 2)
conf1 = {
    'model1':{
        'model': m1,
        'criterion': nn.CrossEntropyLoss(),
        'optimizer': torch.optim.Adam(m1.parameters())
    }
}

m2 = ExModel(10, 10, 2)
conf2 = {
    'model2':{
        'model': m2,
        'criterion': nn.CrossEntropyLoss(),
        'optimizer': torch.optim.Adam(m2.parameters())
    }
}

m3 = ExModel(1000, 10, 2)
conf3 = {
    'model3':{
        'model': m3,
        'criterion': nn.CrossEntropyLoss(),
        'optimizer': torch.optim.Adam(m3.parameters())
    }
}

if __name__ == '__main__':
    rack = trainRack.Rack()

    rack.add_arguments(arg_dict)
    rack.parse_args()

    rack.add_model_configs({**conf1,**conf2,**conf3})

    rack.set_dataset(RandDataset())

    rack.launch()
