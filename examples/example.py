import torch.nn as nn
import torch
from torch.utils import data

from trainRack import Rack

class ExModel(nn.Module):
    def __init__(self,hs,ins,nc):
        super(ExModel,self).__init__()
        self.rec = nn.GRU(input_size=ins,hidden_size=hs,batch_first=True)
        self.cla = nn.Linear(hs,nc)

    def forward(self, input, hx=None):
        out, hn = self.rec(input)
        return nn.Softmax(dim=-1)(self.cla(hn[-1,:,:]))

model = ExModel(20, 10, 2)


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
optimizer_class = torch.optim.Adam
criterion = nn.CrossEntropyLoss()

if __name__ == '__main__':
    rack = Rack()

    rack.add_arguments(arg_dict)
    rack.parse_args()

    rack.set_model(model)
    rack.set_optimizer(optimizer_class)
    rack.set_criterion(criterion)
    rack.set_dataset(RandDataset())

    rack.launch()
