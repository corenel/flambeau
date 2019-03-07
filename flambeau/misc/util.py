import pickle
import warnings

import horovod.torch as hvd
import numpy as np
import torch


def save_pickle(obj, path):
    """
    Save object to path

    :param obj: object to save
    :param path: path to pickle file
    :type path: str
    """
    with open(path, 'wb') as f:
        pickle.dump(obj, f)


def load_pickle(path):
    """
    Load object from path

    :param path: path to pickle file
    :type path: str
    :return: object
    """
    with open(path, 'rb') as f:
        obj = pickle.load(f)
    return obj


def enable_cudnn_autotuner():
    """
    Enable cuDNN auto-tuner to accelerate convolution
    """
    torch.backends.cudnn.benchmark = True


def manual_seed(seed, deterministic=False):
    """
    Set manual random seed

    :param seed: random seed
    :type seed: int
    :param deterministic: whether or not to use deterministic cuDNN setting
    :type deterministic: bool
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')


class AverageMeter(object):
    def __init__(self, name, distributed=False):
        """
        Computes and stores the average and current value

        :param name:
        :type name:
        :param distributed:
        :type distributed:
        """
        self.name = name
        self.sum = torch.Tensor(0.)
        self.count = torch.Tensor(0.)
        self.distributed = distributed

    def reset(self):
        self.sum = torch.Tensor(0.)
        self.count = torch.Tensor(0.)

    def update(self, val):
        if isinstance(val, float):
            val = torch.Tensor(val)
        assert isinstance(val, torch.Tensor)
        if self.distributed:
            val = hvd.allreduce(val.detach().cpu(), name=self.name)
        else:
            val = val.detach().cpu()
        self.sum += val
        self.count += 1

    @property
    def avg(self):
        return self.sum / self.count


def init_linear(linear):
    torch.nn.init.xavier_normal(linear.weight)
    linear.bias.data.zero_()


def init_conv(conv, glu=True):
    torch.nn.init.kaiming_normal(conv.weight)
    if conv.bias is not None:
        conv.bias.data.zero_()
