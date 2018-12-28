import pickle

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


