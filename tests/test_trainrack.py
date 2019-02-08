import subprocess
import shutil
import pytest

epochs = 4
batch_size = 32
test_epoch = 2
num_workers = 6
lr = 0.001
num_classes = 2


def run_training(save_mode='last', kfold=0, validation=0, device = None):
    cmd = ['python', 'guinea_pig.py']
    cmd.extend(['--num_classes', num_classes])

    cmd.extend(['--epochs', epochs])
    cmd.extend(['--batch_size', batch_size])
    cmd.extend(['--test_epoch', test_epoch])
    cmd.extend(['--num_workers', num_workers])
    cmd.extend(['--lr', lr])

    if device is not None:
        cmd.extend(['--device', device])

    cmd.extend(['--validation', validation])
    cmd.extend(['--kfold', kfold])

    if save_mode == 'last':
        cmd.extend(['--save_last', 1])

    if save_mode == 'best':
        cmd.extend(['--save_best', 1])

    if save_mode == 'all':
        cmd.extend(['--save_all', 1])

    cmd = list(map(str, cmd))

    r = subprocess.call(cmd)

    shutil.rmtree('./results')

    return r




save_confs = ['last', 'all', 'best']
kfconfs = [0, 3]
val_confs = [0, 1]

confs = []
for s in save_confs:
    for k in kfconfs:
        for v in val_confs:
            confs.append('{}_kfold{}_validation{}'.format(s, k, v))


@pytest.mark.parametrize("conf", confs)
def test_running(conf):
    l = conf.split('_')
    s = l[0]
    k = l[1][-1]
    v = l[2][-1]
    assert run_training(s, k, v) == 0


def test_cpu():
    assert run_training(device='cpu') == 0

def test_gpu():
    assert run_training(device='cuda') == 0



