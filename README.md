# IZITORCH

*A high-level wrapper for PyTorch.* 

The main tool is the trainRack which implements a Rack class to abstract most of  the common 
steps involved in training deep learning models on PyTorch. This class allows one to produce a **training script in just
a few steps**, while keeping a good level of customization. This class also supports **multi-model training** i.e. 
training multiple models at the same time on one device and on the same dataset (provided that they fit in memory).

## Installation

Clone repo and run:

`pip install -e path/to/izitorch`

Requirements:
- torch
- torchnet
- numpy
- scipy
- scikit-learn

## Getting started
### The trainRack

The trainRack allows one to produce a customizable training script in only two steps: 
provide a dataset, and a model configuration.

The Rack class then takes care of the common mecanisms of training (forward pass, back-propagation, model testing, 
checkpoints etc ...).

The basic structure of a script using the trainRack should thus be:

```Python3
from izitorch.trainRack import Rack, ModelConfig

from mymodels import Model
from mydatasets import Dataset

rack = Rack()

rack.parse_args()

m = Model()

conf = ModelConfig(name='model1', model=m, criterion=nn.CrossEntropyLoss(), 
    optimizer=torch.optim.Adam(m.parameters(), lr = rack.args.lr))

rack.add_model_configs([conf])

rack.set_dataset(Dataset())

rack.launch()

```
