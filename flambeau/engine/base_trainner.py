from abc import abstractmethod

import horovod.torch as hvd
from tensorboardX import SummaryWriter

from .base_engine import BaseEngine


def get_graph(graph):
    """
    Get correct graph due to data parallel

    :param graph: network graph
    :type graph: torch.nn.Module
    :return: correct graph
    :rtype: torch.nn.Module
    """
    if hasattr(graph, 'module'):
        # fix DataParallel
        return graph.module
    else:
        return graph


class BaseTrainer(BaseEngine):
    def __init__(self, hps, result_subdir,
                 step, epoch, devices, data_device, batch_size,
                 verbose=True):
        """
        Network trainer

        :param hps: hyper-parameters for this network
        :param result_subdir: path to result sub-directory
        :type result_subdir: str
        :param step: global step of model
        :type step: int
        :param epoch: global epoch of model
        :type epoch: int
        :param devices: list of available devices for model running
        :type devices: list
        :param data_device: available device for data loading
        :type data_device: str or int
        :param batch_size: number of inputs in mini-batch
        :type batch_size: int or dict
        :param verbose: whether or not to print running messages
        :type verbose: bool
        """
        super().__init__(verbose)

        # general
        self.hps = hps
        self.result_subdir = result_subdir
        self.distributed = hps.device.distributed.enabled
        # horovod: print logs on the first worker.
        if self.distributed:
            self.verbose = hvd.rank() == 0

        # state
        self.step = step
        self.epoch = epoch
        self.devices = devices
        self.num_device = len(devices)

        # data
        self.data_device = data_device
        self.batch_size = batch_size
        self.num_classes = self.hps.dataset.num_classes

        # logging
        self.is_output_rank = self.verbose

        self.writer = SummaryWriter(logdir=self.result_subdir) if self.is_output_rank else None
        self.interval_scalar = self.hps.optim.interval.scalar
        self.interval_snapshot = self.hps.optim.interval.snapshot

    @abstractmethod
    def train(self):
        pass
