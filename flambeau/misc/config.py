import json
import os
from collections import OrderedDict

import numpy as np
import torch
import yaml
import sys
import pprint


class OrderedEasyDict(OrderedDict):

    def __init__(self, input_dict=None, **kwargs):
        """
        EasyDict that remembers the order

        :param input_dict: given dictionary
        :type input_dict: dict
        """
        super().__init__()
        # initial dict
        if input_dict is None:
            input_dict = {}
        if kwargs:
            input_dict.update(**kwargs)
        # set attributes
        for k, v in input_dict.items():
            setattr(self, k, v)
        # set class attributes
        for k in self.__class__.__dict__.keys():
            if not (k.startswith('__') and k.endswith('__')):
                setattr(self, k, getattr(self, k))

    def __setattr__(self, name, value):
        if isinstance(value, (list, tuple)):
            value = [self.__class__(x)
                     if isinstance(x, dict) else x for x in value]
        elif isinstance(value, dict) and not isinstance(value, self.__class__):
            value = self.__class__(value)
        super(OrderedEasyDict, self).__setattr__(name, value)
        super(OrderedEasyDict, self).__setitem__(name, value)

    __setitem__ = __setattr__


def load_profile(file_path,
                 data_root=None,
                 result_dir=None,
                 snapshot=None,
                 gpu=None):
    """
    Load experiment profile as EasyDict

    :param file_path: path to profile
    :type file_path: str
    :param snapshot: path to snapshot file
    :type snapshot: str
    :param data_root: path to data root
    :type data_root: str
    :param result_dir: path to result dir
    :type result_dir: str
    :param gpu: the id of gpu to use
    :type gpu: int
    :return: hyper-parameters
    :rtype: OrderedEasyDict
    """
    # check output directory
    assert os.path.exists(file_path)
    # check extension of profile file
    file_type = os.path.splitext(file_path)[1]
    assert file_type in ('.json', '.yml', '.yaml')
    # load profile
    with open(file_path) as f:
        if file_type == '.json':
            profile_dict = ordered_json_load(f)
        else:
            profile_dict = ordered_yaml_load(f, yaml.SafeLoader)
        hps = OrderedEasyDict(profile_dict)

    # set random seed
    manual_seed(hps.ablation.seed)

    # override attributes with given arguments
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


def dump_profile(profile, output_dir, file_type='yaml'):
    """
    Load experiment profile as EasyDict

    :param profile: profile
    :type profile: OrderedEasyDict
    :param output_dir: path to output directory
    :type output_dir: str
    :param file_type: type of profile file
    :type file_type: str
    """
    # check output directory
    assert os.path.exists(output_dir)
    # check extension of profile file
    assert file_type in ('json', 'yml', 'yaml')
    # dump profile
    output_path = os.path.join(output_dir, 'config.{}'.format(file_type))
    with open(output_path, 'w') as f:
        if file_type == 'json':
            json.dump(profile, f,
                      indent=2,
                      separators=(',', ': '))
        else:
            profile_orderd_dict = to_ordered_dict(profile)
            ordered_yaml_dump(data=profile_orderd_dict,
                              stream=f,
                              dumper=yaml.SafeDumper)


def print_profile(profile):
    """
    Print profile prettily

    :param profile: profile dictionary
    :type profile: OrderedEasyDict
    """
    pprint.pprint(profile)


def manual_seed(seed):
    """
    Set manual random seed

    :param seed: random seed
    :type seed: int
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)


def ordered_yaml_load(stream,
                      loader=yaml.Loader,
                      object_pairs_hook=OrderedDict):
    # check Python version
    insertion_order_preservation_in_dict = sys.version_info >= (3, 7)
    # load
    if insertion_order_preservation_in_dict:
        return yaml.load(stream, loader)
    else:
        class OrderedLoader(loader):
            pass

        def construct_mapping(node_loader, node):
            node_loader.flatten_mapping(node)
            return object_pairs_hook(node_loader.construct_pairs(node))

        OrderedLoader.add_constructor(
            yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
            construct_mapping)
        return yaml.load(stream, OrderedLoader)


def ordered_yaml_dump(data,
                      stream=None,
                      dumper=yaml.Dumper,
                      **kwargs):
    # check Python version
    insertion_order_preservation_in_dict = sys.version_info >= (3, 7)
    # dump
    if False and insertion_order_preservation_in_dict:
        yaml.dump(data, stream, dumper, **kwargs)
    else:
        class OrderedDumper(dumper):
            pass

        def _dict_representer(dict_dumper, dict_data):
            return dict_dumper.represent_mapping(
                yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
                dict_data.items())

        OrderedDumper.add_representer(OrderedDict, _dict_representer)
        return yaml.dump(data, stream, OrderedDumper, **kwargs)


def ordered_json_load(fp):
    # check Python version
    insertion_order_preservation_in_dict = sys.version_info >= (3, 7)
    # load
    if insertion_order_preservation_in_dict:
        return json.load(fp)
    else:
        return json.load(fp, object_pairs_hook=OrderedDict)


def to_ordered_dict(input_ordered_dict):
    return json.loads(json.dumps(input_ordered_dict), object_pairs_hook=OrderedDict)
