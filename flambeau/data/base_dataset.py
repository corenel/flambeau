from torch.utils.data import Dataset


class BaseDataset(Dataset):
    def __init__(self,
                 root,
                 transform=None,
                 target_transform=None):
        """
        Base class for custom dataset

        :param root: path to data root
        :type root: str
        :param transform: transform applied to data
        :type transform: object
        :param target_transform: transform applied to target
        :type target_transform: object
        """
        super().__init__()
        self.root = root
        self.transform = transform
        self.target_transform = target_transform

    def __repr__(self):
        fmt_str = '[Dataset] ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of data points: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(
            tmp,
            self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(
            tmp,
            self.target_transform.__repr__().replace('\n',
                                                     '\n' + ' ' * len(tmp)))
        return fmt_str
