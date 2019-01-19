import pickle
import warnings

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
