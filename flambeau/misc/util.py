import json
import os
import pickle

import numpy as np
import torch

from .ordered_easydict import OrderedEasyDict


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


def manual_seed(seed):
    """
    Set manual random seed

    :param seed: random seed
    :type seed: int
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)


def load_profile(filepath,
                 data_root=None,
                 result_dir=None,
                 snapshot=None,
                 gpu=None):
    """
    Load experiment profile as EasyDict

    :param filepath: path to profile
    :type filepath: str
    :param snapshot: path to snapshot file
    :type snapshot: str
    :param data_root: path to data root
    :type data_root: str
    :param result_dir: path to result dir
    :type result_dir: str
    :param gpu: the id of gpu to use
    :type gpu: int
    :return: hyper-parameters
    :rtype: EasyDict
    """
    hps = None
    if os.path.exists(filepath):
        with open(filepath) as f:
            hps = OrderedEasyDict(json.load(f, object_pairs_hook=OrderedDict))

    manual_seed(hps.ablation.seed)

    if snapshot is not None:
        hps.general.pre_trained = snapshot
        hps.general.warm_start = True
    if data_root is not None:
        hps.dataset.root = data_root
    if result_dir is not None:
        hps.general.result_dir = result_dir
    if gpu is not None:
        hps.device.graph = ['cuda:{}'.format(gpu)]
        hps.device.data = ['cuda:{}'.format(gpu)]

    return hps



