import os
import re
from abc import abstractmethod
from functools import partial

import torch
import torch.distributed as dist
from flambeau.misc import saver
from flambeau.network import lr_scheduler
from flambeau.network.optimizer import AdamW

from .base_engine import BaseEngine


def devices_to_string(devices):
    """
    Format device list to string

    :param devices: list of devices
    :type devices: int or str or list
    :return: string of device list
    :rtype: str
    """
    if isinstance(devices, str):
        return devices
    if isinstance(devices, int):
        devices = [devices]
    return ', '.join(['cuda:{}'.format(d) for d in devices])


def get_devices(devices):
    """
    Get devices for running model

    :param devices: list of devices from profile
    :type devices: list
    :return: list of usable devices according to desired and available hardware
    :rtype: list[str]
    """

    def parse_cuda_device(device):
        """
        Parse device into device id

        :param device: given device
        :type device: str or int
        :return: device id
        :rtype: int
        """
        origin = str(device)
        if isinstance(device, str) and re.search(r'cuda:([\d]+)', device):
            device = int(re.findall(r'cuda:([\d]+)', device)[0])
        if isinstance(device, int):
            if 0 <= device <= torch.cuda.device_count() - 1:
                return device
        print('[Builder] Incorrect device "{}"'.format(origin))
        return

    use_cpu = any([d.find('cpu') >= 0 for d in devices if isinstance(d, str)])
    use_cuda = any(
        [isinstance(d, int) or (isinstance(d, str) and d.find('cuda') >= 0)
         for d in devices])
    assert not (use_cpu and use_cuda), 'CPU and GPU cannot be mixed.'

    if use_cuda:
        devices = [parse_cuda_device(d) for d in devices]
        devices = [d for d in devices if d is not None]
        if len(devices) == 0:
            print('[Builder] No available GPU found, use CPU only')
            devices = ['cpu']

    return devices


class BaseBuilder(BaseEngine):
    optimizer_dict = {
        'adam': lambda params, **kwargs: torch.optim.Adam(params, **kwargs),
        'adamw': lambda params, **kwargs: AdamW(params, **kwargs),
        'adamax': lambda params, **kwargs: torch.optim.Adamax(params, **kwargs)
    }

    def __init__(self, hps, gpu=None, ngpus_per_node=None, verbose=True):
        """
        Network builder

        :param hps: hyper-parameters for this network
        :param verbose: whether or not to print running messages
        :type verbose: bool
        """
        super().__init__(verbose)
        self.hps = hps

        # distributed training
        self.distributed = hps.device.distributed.enabled
        if isinstance(gpu, str) and re.search(r'cuda:([\d]+)', gpu):
            gpu = int(re.findall(r'cuda:([\d]+)', gpu)[0])
        self.given_gpu_id = gpu
        self.ngpus_per_node = ngpus_per_node
        if self.distributed:
            hps.device.graph = [gpu]
            hps.device.data = [gpu]
            if hps.device.distributed.dist_url == "env://" and hps.device.distributed.rank == -1:
                hps.device.distributed.rank = int(os.environ["RANK"])
            if hps.device.distributed.multiprocessing_distributed:
                # For multiprocessing distributed training, rank needs to be the
                # global rank among all the processes
                hps.device.distributed.rank = hps.device.distributed.rank * ngpus_per_node + gpu
            dist.init_process_group(backend=hps.device.distributed.dist_backend,
                                    init_method=hps.device.distributed.dist_url,
                                    world_size=hps.device.distributed.world_size,
                                    rank=hps.device.distributed.rank)

    @abstractmethod
    def build(self):
        pass

    def _make_optimizer(self, model,
                        optimizer_name=None,
                        optimizer_args=None):
        """
        Make optimizer for parameters of given model

        If you need to move a model to GPU via model.cuda(),
        please do so before constructing optimizers for it.

        :param model: given model
        :type model: torch.nn.Module
        :param optimizer_name: name of optimizer
        :type optimizer_name: str
        :param optimizer_args: arguments of optimizer
        :type optimizer_args: dict
        :return: optimzier for given model
        :rtype: torch.optim.Optimizer
        """
        if optimizer_name is None:
            optimizer_name = self.hps.optim.optimizer.name.lower()
        if optimizer_args is None:
            optimizer_args = self.hps.optim.optimizer.args.copy()
        assert optimizer_name in self.optimizer_dict.keys(), \
            "Unsupported optimizer: {}".format(optimizer_name)

        optimizer = self.optimizer_dict[optimizer_name](
            filter(lambda p: p.requires_grad, model.parameters()),
            **optimizer_args)

        self._print('Use optimizer for {}: {}'.format(
            model.__class__.__name__,
            optimizer_name))
        for k, v in optimizer_args.items():
            self._print('    {}: {}'.format(k, v), show_name=False)

        return optimizer

    def _make_lr_scheduler_neo(self,
                               model,
                               optimizer,
                               scheduler_name=None,
                               scheduler_args=None):
        """
        Make learning rate scheduler

        :param model: given model
        :type model: torch.nn.Module
        :param: optimzier for given model
        :type: torch.optim.Optimizer
        :param scheduler_name: name of lr scheduler
        :type scheduler_name: str
        :param scheduler_args: arguments of lr scheduler
        :type scheduler_args: dict
        :return: lr scheduler
        :rtype: function
        """
        if scheduler_name is None:
            scheduler_name = self.hps.optim.lr_scheduler.name.lower()
        if scheduler_args is None:
            scheduler_args = self.hps.optim.lr_scheduler.args.copy()
        assert scheduler_name in self.lr_scheduler_dict.keys(), \
            "Unsupported lr scheduler: {}".format(scheduler_name)

    def _make_lr_scheduler(self,
                           model,
                           scheduler_name=None,
                           scheduler_args=None,
                           base_lr=None):
        """
        Make learning rate scheduler

        :param model: given model
        :type model: torch.nn.Module
        :param scheduler_name: name of lr scheduler
        :type scheduler_name: str
        :param scheduler_args: arguments of lr scheduler
        :type scheduler_args: dict
        :param base_lr: init lr
        :type base_lr: float
        :return: lr scheduler
        :rtype: function
        """
        if scheduler_name is None:
            scheduler_name = self.hps.optim.lr_scheduler.name.lower()
        if scheduler_args is None:
            scheduler_args = self.hps.optim.lr_scheduler.args.copy()
        assert scheduler_name in lr_scheduler.lr_scheduler_dict.keys(), \
            "Unsupported lr scheduler: {}".format(scheduler_name)

        if 'base_lr' not in scheduler_args:
            if base_lr is not None:
                scheduler_args['base_lr'] = base_lr
            else:
                scheduler_args['base_lr'] = self.hps.optim.optimizer.args['lr']
        if 'warmup_epochs' in scheduler_args:
            del scheduler_args['warmup_epochs']

        scheduler = partial(lr_scheduler.lr_scheduler_dict[scheduler_name],
                            **scheduler_args)

        self._print('Use lr scheduler for {}: {}'.format(
            model.__class__.__name__,
            scheduler_name))
        for k, v in scheduler_args.items():
            self._print('    {}: {}'.format(k, v), show_name=False)

        return scheduler

    def _make_devices(self):
        devices = get_devices(self.hps.device.graph)
        data_device = get_devices(self.hps.device.data)[0]

        if 'cpu' in devices:
            data_device = 'cpu'

        self._print('Use {} for model running and {} for data loading'.format(
            devices_to_string(devices),
            devices_to_string(data_device)))

        return devices, data_device

    def _get_result_subdir(self, training=True):
        result_subdir = None
        if self.hps.general.warm_start and self.hps.general.resume_run_id is not None:
            result_subdir = saver.locate_result_subdir(
                self.hps.general.result_dir, self.hps.general.resume_run_id)

        if training and result_subdir is None:
            result_subdir = saver.create_result_subdir(
                self.hps.general.result_dir,
                desc=self.hps.experiment,
                profile=self.hps,
                copy=True)
        return result_subdir

    def _load_state(self, graph, result_subdir, training=True):
        state = None
        if self.hps.general.warm_start:
            step_or_model_path = None
            if os.path.exists(self.hps.general.pre_trained):
                step_or_model_path = self.hps.general.pre_trained
            elif self.hps.general.resume_step == 'best':
                step_or_model_path = os.path.join(
                    result_subdir, saver.get_best_model_name())
            elif self.hps.general.resume_step == 'latest':
                step_or_model_path = os.path.join(
                    result_subdir, saver.get_latest_model_name(result_subdir))
            elif self.hps.general.resume_step != '':
                step_or_model_path = os.path.join(
                    result_subdir,
                    saver.get_model_name(int(self.hps.general.resume_step)))

            if step_or_model_path is not None:
                state = saver.load_snapshot(graph, step_or_model_path)
                epoch = state['epoch']
                self._print('Resume from step {}'.format(epoch))

        if not training and state is None:
            self._print('No pre-trained model for inference')

        return state

    def _move_model_to_device(self, model, devices=('cpu',)):
        """
        Move model to correct devices

        :param model: given model
        :type model: torch.nn.Module
        :param devices: list of available devices for model running
        :type devices: list
        :return: model in correct device
        :rtype: torch.nn.Module
        """
        if 'cpu' in devices:
            model = model.cpu()
        elif self.distributed:
            if self.given_gpu_id is not None:
                # For multiprocessing distributed, DistributedDataParallel constructor
                # should always set the single device scope, otherwise,
                # DistributedDataParallel will use all available devices.
                torch.cuda.set_device(self.given_gpu_id)
                model.cuda(self.given_gpu_id)
                model = torch.nn.parallel.DistributedDataParallel(
                    module=model,
                    device_ids=[self.given_gpu_id])
            elif devices:
                # Use specific gpu devices in DistributedDataParallel
                model.cuda()
                model = torch.nn.parallel.DistributedDataParallel(
                    module=model,
                    device_ids=devices)
            else:
                # DistributedDataParallel will divide and allocate batch_size to all
                # available GPUs if device_ids are not set
                model.cuda()
                model = torch.nn.parallel.DistributedDataParallel(model)
        elif self.given_gpu_id is not None:
            # Use given single gpu in DataParallel
            torch.cuda.set_device(self.given_gpu_id)
            model = model.cuda(self.given_gpu_id)
        elif devices:
            # Use specific gpu devices in DataParallel
            model.cuda()
            model = torch.nn.parallel.DataParallel(
                module=model,
                device_ids=devices)
        else:
            # DataParallel will divide and allocate batch_size to all available GPUs
            model = torch.nn.DataParallel(model).cuda()

        return model

    def _make_batch_size(self, training=True):
        batch_size = self.hps.optim.batch_size.train if training else self.hps.optim.batch_size.eval
        if self.distributed and self.given_gpu_id is not None:
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            batch_size = int(batch_size / self.ngpus_per_node)
            self.hps.dataset.num_workers = int(self.hps.dataset.num_workers / self.ngpus_per_node)
        self._print('Use batch size : {}'.format(batch_size))
        return batch_size
