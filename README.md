# IZITORCH

*A high-level wrapper for PyTorch.* 

The main module is  ```trainRack ``` which implements a Rack class to abstract most of  the common 
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

The Rack class then takes care of the common mechanisms of training (forward pass, back-propagation, model testing, 
checkpoints etc ...).

The basic structure of a script using the trainRack should thus be:

```Python3
import torch
from izitorch.trainRack import Rack, ModelConfig

from mymodels import Model
from mydatasets import Dataset

rack = Rack()

rack.parse_args()

m = Model()

conf = ModelConfig(name='model1', model=m, criterion=torch.nn.CrossEntropyLoss(), 
    optimizer=torch.optim.Adam(m.parameters(), lr = rack.args.lr))

rack.add_model_configs([conf])

rack.set_dataset(Dataset())

rack.launch()

```


### Rack parameters
Though it manages all the training mechanics under the hood, the Rack is customizable with the parameters menu. 

#### Default parameters
The following default list of parameters is implemented in the Rack class:
- device: device to be used for computation (cpu/cuda)
- res_dir: path to the directory where the checkpoints should be stored
- rdm_seed: Random seed for dataset partitioning
- dataset: Path to dataset file/folder (if required)
- num_classes: number of classes (in case of classification problem)
- num_workers: number of workers for DataLoader
- pin_memory: whether to use pin memory for DataLoader
- train_ratio: ratio of total number of samples used as train set (when no k-fold)
- kfold: If non zero, number of folds to execute
- validation: Whether to use a separate validation test to test each epoch
- save_last: Save only the last epoch's weights
- save_all: Save all the epochs' weights
- save_best: Save only the best epoch's weights
- metric_best: Metric to use to assess best epoch (loss/acc/IoU)
- epochs: total number of epochs of training
- batch_size: batch size
- lr: learning rate
- test_epoch: number of epochs between tests
- display_step: number of training steps between two displays of progress
- shuffle: shuffle dataset
- grad_clip: If non zero, gradients will be clipped at this value

#### Custom parameters
Additional custom arguments can be added to the menu using the method ```rack.add_arguments(args)``` where args is a 
dictionary specifying the names, types, and default values of the additional arguments. 

#### Passing the parameters
The values chosen for the parameters can be passed in two ways: 
1. From the command line, when the script is called\
```python train_model.py --batch_size 32 --lr 0.005 --save_all 1 --epochs 1000```\
A Rack instance comes with an argparse menu. Calling the ```rack.parse_args()``` method at the beginning 
of your script will retrieve the parameters passed in the command line call and store them in the ```rack.args```
 attribute. All the parameters that are not specified will be set at their default value (see default_config.json).

2. Directly inside the script\
The parameters can also be set inside the script using the ```rack.set_args(args)``` method, where args is a dictionnary 
specifying the values of the parameters. 

### Rack outputs
The output of the Rack in the directory specified by ```--res_dir``` comprises of:

- A configuration file conf.json keeping track of the parameters passed to the training script.
- The checkpoints of the model (weights, and optimizer state)
- A json trainlog that is updated at each epoch
- A pickle file with the final performance metrics (e.g. the final confusion matrix in case of classification)


### Model Configuration and multi-model training

Model configurations should be formatted using the ```trainRack.ModelConfig``` class. As seen in the example above,
a configuration is defined by a name, a model instance, a criterion and an optimizer. 
The Rack is designed to support multi-model training, i.e. one can attach a list of ModelConfigs to it with the
 ```rack.add_model_configs(confs)``` method. Provided that they all fit in memory and that they are trained on the same
 dataset, the Rack will train all the models simultaneously. 
 
 **Important note** the computations for each model are not parallelized. For each batch of the DataLoader, the Rack will
 sequentially execute the forward-backward process for each model. This can be useful for small models when data loading
  is the most time consuming step, and you want to make the most of it! 


### Dataset

The dataset attached to the rack via  ```rack.set_dataset(dataset)``` should inherit 
from the  ```torch.utils.data.Dataset``` class. 
