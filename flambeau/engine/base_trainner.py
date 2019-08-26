import collections
import os
from abc import abstractmethod

from comet_ml import Experiment
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


def flatten_dict(d, parent_key='', sep='-'):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, collections.MutableMapping):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


class BaseTrainer(BaseEngine):
    def __init__(self,
                 hps,
                 result_subdir,
                 step,
                 epoch,
                 devices,
                 data_device,
                 batch_size,
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
        if hps.logging.tensorboard.enabled:
            self.writer = SummaryWriter(
                logdir=self.result_subdir) if self.is_output_rank else None
        if hps.logging.comet.enabled:
            self.experiment = Experiment(
                project_name=hps.logging.comet.project_name,
                workspace=hps.logging.comet.workspace
            ) if self.is_output_rank else None
            if self.is_output_rank and self.experiment.alive is False:
                raise RuntimeError('Something went wrong w/ comet.ml')
        self.log_profile(self.hps)
        self.interval_scalar = self.hps.logging.interval.scalar
        self.interval_snapshot = self.hps.logging.interval.snapshot

    @abstractmethod
    def train(self):
        pass

    def log_profile(self, hps):
        if self.hps.logging.comet.enabled and self.is_output_rank:
            self.experiment.log_parameters(flatten_dict(hps))
            self.experiment.log_parameter('general-result_subdir',
                                          self.result_subdir)
            self.experiment.set_name(
                self.result_subdir.replace(
                    self.hps.general.result_dir
                    if self.hps.general.result_dir.endswith('/') else
                    self.hps.general.result_dir + '/', ''))

    def log_scalar(self, name, value, step):
        if self.is_output_rank:
            if self.hps.logging.tensorboard.enabled:
                self.writer.add_scalar(name, value, step)
            if self.hps.logging.comet.enabled:
                self.experiment.log_metric(name, value, step)

    def log_close(self):
        if self.hps.logging.tensorboard.enabled and self.is_output_rank:
            self.writer.export_scalars_to_json(
                os.path.join(self.result_subdir, 'all_scalars.json'))
            self.writer.close()
