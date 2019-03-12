from collections.abc import Iterable

import torchvision.transforms.functional as F
from PIL import Image


class CenterResize:

    def __init__(self, size, interpolation=Image.BILINEAR):
        assert isinstance(size, int) or \
               (isinstance(size, Iterable) and len(size) == 2)
        self.size = size
        self.interpolation = interpolation

    def __call__(self, img):
        """
        Crop PIL image at the center

        :param img: image to crop
        :type img: PIL.Image.Imgae
        :return: cropped image
        :rtype: PIL.Image.Image
        """

        size = min(*img.size)
        img = F.center_crop(img, size)
        img = F.resize(img, self.size, self.interpolation)
        assert img.size == (self.size, self.size), \
            '{} != {}'.format(img.size, (self.size, self.size))
        return img

    def __repr__(self):
        return self.__class__.__name__
