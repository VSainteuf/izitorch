import torch.nn as nn
import torch
from torch.utils import data

from trainRack import Rack



model = nn.GRU(10,20,1)

class RandDataset(data.Dataset):
    def __init__(self,nsamp=10000,seqlen=5,nfeat=10,nclass=2):
        super(RandDataset, self).__init__()
        self.nsamp = nsamp
        self.nfeat = nfeat
        self.data = torch.randn(nsamp,seqlen,nfeat+nclass)
    def __getitem__(self, item):
        return self.data[item,:,:nfeat],self.data[item,:,nfeat:]

    def __len__(self):
        return self.nsamp


arg_dict = {'extra_feature':{'default':2,'type':int}}
optimizer_class = torch.optim.Adam
criterion = nn.CrossEntropyLoss()






if __name__=='__main__':
    rack = Rack()

    rack.add_arguments(arg_dict)
    rack.parse_args()


    rack.set_model(model)
    rack.set_optimizer(optimizer_class)
    rack.set_criterion(criterion)
    rack.dataset = RandDataset()

    rack.launch()

