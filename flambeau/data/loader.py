import horovod.torch as hvd
import torch
from PIL import Image
from torch.utils.data import DataLoader
from torchvision.transforms import transforms

from flambeau.engine import BaseEngine
from . import transforms as custom_transforms


def generate_transform(transform_dict):
    transform_list = []
    for k, v in transform_dict.items():
        if k.lower() == 'resize':
            transform_list.append(transforms.Resize(v))
        elif k.lower() == 'resize_nearest':
            transform_list.append(transforms.Resize(v, interpolation=Image.NEAREST))
        elif k.lower() == 'crop':
            transform_list.append(transforms.RandomCrop(v))
        elif k.lower() == 'resized_crop':
            assert len(v) == 4
            h, w, min_scale = v
            transform_list.append(
                transforms.RandomResizedCrop((h, w), (min_scale, 1.)))
        elif k.lower() == 'center_crop':
            transform_list.append(
                transforms.CenterCrop(v))
        elif k.lower() == 'center_resize':
            transform_list.append(
                custom_transforms.CenterResize(v))
        elif k.lower() == 'resize_center':
            transform_list.append(
                custom_transforms.CenterResize(v, resize_first=True))
        elif k.lower() == 'h_flip':
            transform_list.append(transforms.RandomHorizontalFlip())
        elif k.lower() == 'rotate':
            transform_list.append(transforms.RandomRotation(v))
        elif k.lower() == 'color_jitter':
            assert len(v) == 4
            brightness, contrast, saturation, hue = v
            transform_list.append(
                transforms.ColorJitter(brightness, contrast, saturation, hue))
        elif k.lower() == 'to_tensor':
            transform_list.append(transforms.ToTensor())
        elif k.lower() == 'normalize':
            if v is None:
                norm_value = ([0.485, 0.456, 0.406],
                              [0.229, 0.224, 0.225])
            elif len(v) == 6:
                assert len(v) == 6
                norm_value = ([v[0], v[1], v[2]],
                              [v[3], v[4], v[5]])
            elif len(v) == 2:
                norm_value = ([v[0]],
                              [v[1]])
            else:
                raise RuntimeError('Invalid arguments for normalize transform')

            transform_list.append(
                transforms.Normalize(mean=norm_value[0], std=norm_value[1]))
        else:
            print('Unsupported data transform type: {}'.format(k))

    return transforms.Compose(transform_list)


def make_data_loader(dataset, batch_size, shuffle=True, num_workers=8, distributed=False):
    if distributed:
        # horovod: use DistributedSampler to partition data among workers.
        # Manually specify `num_replicas=hvd.size()` and `rank=hvd.rank()`.
        data_sampler = torch.utils.data.distributed.DistributedSampler(
            dataset, num_replicas=hvd.size(), rank=hvd.rank())
    else:
        data_sampler = None

    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(data_sampler is None),
        num_workers=num_workers,
        pin_memory=True,
        sampler=data_sampler,
        drop_last=True
    )
    return data_loader


class DatasetLoader(BaseEngine):

    def __init__(self, hps, dataset_dict=None):
        """
        Loader for custom dataset

        :param hps: profile
        :param dataset_dict: dict of custom dataset
        :type dataset_dict: dict
        """
        super().__init__()
        # load profile
        self.hps = hps

        # horovod: print logs on the first worker.
        self.distributed = hps.device.distributed.enabled
        if self.distributed:
            self.verbose = hvd.rank() == 0

        # check support of dataset
        self.name = hps.dataset.problem.lower()
        if dataset_dict is not None:
            self.dataset_dict = dataset_dict
        else:
            self.dataset_dict = {}
        assert self.name in self.dataset_dict.keys(), \
            'Unsupported dataset: {}'.format(self.name)

        # transform
        transform_dict = hps.dataset.transforms
        self.train_transforms = None
        if transform_dict is not None:
            if 'train' in transform_dict or 'valid' in transform_dict:
                if 'train' in transform_dict:
                    self.train_transforms = generate_transform(
                        hps.dataset.transforms.train)
                if 'valid' in transform_dict:
                    self.valid_transforms = generate_transform(
                        hps.dataset.transforms.valid)
                else:
                    self.valid_transforms = self.train_transforms
            else:
                self.train_transforms = generate_transform((hps.dataset.transforms))
                self.valid_transforms = self.train_transforms

        if hasattr(hps.dataset, 'args'):
            self.args = hps.dataset.args
        else:
            self.args = {}

    def load(self, split=False, split_ratio=0.8, get_loader=False):
        """
        Load dataset

        :param split: whether or not to split dataset into train and valid parts
        :type split: bool
        :param split_ratio: split ratio of training set
        :type split_ratio: float
        :param get_loader: whether ot not to return a data loader
        :type get_loader: bool
        :return: desired dataset or data loader
        :rtype: torch.utils.data.Dataset or torch.utils.data.DataLoader
        """
        if split:
            # get dataset
            if self.args is not None:
                full_dataset = self.dataset_dict[self.name](
                    self.hps.dataset.root, **self.args)
            else:
                full_dataset = self.dataset_dict[self.name](
                    self.hps.dataset.root)
            self._print(full_dataset)

            # split dataset
            sp = int(split_ratio * len(full_dataset))
            train_dataset, valid_dataset = torch.utils.data.random_split(
                full_dataset, [sp, len(full_dataset) - sp])

            # apply transforms
            train_dataset.dataset.transform = self.train_transforms
            valid_dataset.dataset.transform = self.valid_transforms
            self._print('[Dataset] Train transforms: ')
            self._print('\t {}'.format(self.train_transforms.__repr__()))
            self._print('[Dataset] Valid transforms: ')
            self._print('\t {}'.format(self.valid_transforms.__repr__()))

            # get data loader
            if not get_loader:
                return train_dataset, valid_dataset
            else:
                train_data_loader = self._make_data_loader(train_dataset)
                valid_data_loader = self._make_data_loader(valid_dataset)
                return train_data_loader, valid_data_loader
        else:
            # get dataset
            if self.args is not None:
                full_dataset = self.dataset_dict[self.name](
                    self.hps.dataset.root,
                    transform=self.train_transforms,
                    **self.args)
            else:
                full_dataset = self.dataset_dict[self.name](
                    self.hps.dataset.root,
                    transform=self.train_transforms)
            self._print(full_dataset)
            if not get_loader:
                return full_dataset
            else:
                return self._make_data_loader(full_dataset)

    def _make_data_loader(self, dataset):
        return make_data_loader(
            dataset,
            batch_size=self.hps.optim.batch_size.train,
            num_workers=self.hps.dataset.num_workers,
            distributed=self.hps.device.distributed.enabled)
