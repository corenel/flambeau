import torch
import torch.nn.functional as F


def reduce_mean(tensor, dim=None, keepdim=False, out=None):
    """
    Returns the mean value of each row of the input tensor in the given dimension dim.

    Support multi-dim mean

    :param tensor: the input tensor
    :type tensor: torch.Tensor
    :param dim: the dimension to reduce
    :type dim: int or list[int]
    :param keepdim: whether the output tensor has dim retained or not
    :type keepdim: bool
    :param out: the output tensor
    :type out: torch.Tensor
    :return: mean result
    :rtype: torch.Tensor
    """
    # mean all dims
    if dim is None:
        return torch.mean(tensor)
    # prepare dim
    if isinstance(dim, int):
        dim = [dim]
    dim = sorted(dim)
    # get mean dim by dim
    for d in dim:
        tensor = tensor.mean(dim=d, keepdim=True)
    # squeeze reduced dimensions if not keeping dim
    if not keepdim:
        for cnt, d in enumerate(dim):
            tensor.squeeze_(d - cnt)
    if out is not None:
        out.copy_(tensor)
    return tensor


def reduce_sum(tensor, dim=None, keepdim=False, out=None):
    """
    Returns the sum of all elements in the input tensor.

    Support multi-dim sum

    :param tensor: the input tensor
    :type tensor: torch.Tensor
    :param dim: the dimension to reduce
    :type dim: int or list[int]
    :param keepdim: whether the output tensor has dim retained or not
    :type keepdim: bool
    :param out: the output tensor
    :type out: torch.Tensor
    :return: sum result
    :rtype: torch.Tensor
    """
    # summarize all dims
    if dim is None:
        return torch.sum(tensor)
    # prepare dim
    if isinstance(dim, int):
        dim = [dim]
    dim = sorted(dim)
    # get summary dim by dim
    for d in dim:
        tensor = tensor.sum(dim=d, keepdim=True)
    # squeeze reduced dimensions if not keeping dim
    if not keepdim:
        for cnt, d in enumerate(dim):
            tensor.squeeze_(d - cnt)
    if out is not None:
        out.copy_(tensor)
    return tensor


def tensor_equal(a, b, eps=1e-5):
    """
    Compare two tensors

    :param a: input tensor a
    :type a: torch.Tensor
    :param b: input tensor b
    :type b: torch.Tensor
    :param eps: epsilon
    :type eps: float
    :return: whether two tensors are equal
    :rtype: bool
    """
    if a.shape != b.shape:
        return False

    return 0 <= float(torch.max(torch.abs(a - b))) <= eps


def split_channel(tensor, split_type='simple'):
    """
    Split channels of tensor

    :param tensor: input tensor
    :type tensor: torch.Tensor
    :param split_type: type of splitting
    :type split_type: str
    :return: split tensor
    :rtype: tuple(torch.Tensor, torch.Tensor)
    """
    assert len(tensor.shape) == 4
    assert split_type in ['simple', 'cross']

    nc = tensor.shape[1]
    if split_type == 'simple':
        return tensor[:, :nc // 2, ...], tensor[:, nc // 2:, ...]
    elif split_type == 'cross':
        return tensor[:, 0::2, ...], tensor[:, 1::2, ...]


def cat_channel(*args):
    """
    Concatenates channels of tensors

    :return: concatenated tensor
    :rtype: torch.Tensor
    """
    return torch.cat(args, dim=1)


def cat_batch(*args):
    """
    Concatenates batches of tensors

    :return: concatenated tensor
    :rtype: torch.Tensor
    """
    return torch.cat(args, dim=0)


def count_pixels(tensor):
    """
    Count number of pixels in given tensor

    :param tensor: input tensor
    :type tensor: torch.Tensor
    :return: number of pixels
    :rtype: int
    """
    assert len(tensor.shape) == 4
    return int(tensor.shape[2] * tensor.shape[3])


def onehot(y, num_classes):
    """
    Generate one-hot vector

    :param y: ground truth labels
    :type y: torch.Tensor
    :param num_classes: number os classes
    :type num_classes: int
    :return: one-hot vector generated from labels
    :rtype: torch.Tensor
    """
    assert len(y.shape) in [1, 2], "Label y should be 1D or 2D vector"
    y_onehot = torch.zeros(y.shape[0], num_classes).to(y.device, non_blocking=True)
    if len(y.shape) == 1:
        y_onehot = y_onehot.scatter_(1, y.unsqueeze(-1), 1)
    else:
        y_onehot = y_onehot.scatter_(1, y, 1)
    return y_onehot


def de_onehot(y_onehot):
    """
    Convert one-hot vector back to class label

    :param y_onehot: one-hot label
    :type y_onehot: torch.Tensor
    :return: corresponding class
    :rtype: int or Torch.Tensor
    """
    assert len(y_onehot.shape) in [1, 2], \
        "Label y_onehot should be 1D or 2D vector"
    if len(y_onehot.shape) == 1:
        return torch.argmax(y_onehot)
    else:
        return torch.argmax(y_onehot, dim=1)


def resize_feature_map(x, out_shape, interpolate_mode='nearest'):
    """
    Resize feature map into desired shape

    :param x: input feature map
    :type x: torch.Tensor
    :param out_shape: desired tensor shape
    :type out_shape: tuple(int) or list[int]
    :param interpolate_mode: mode for interpolation
    :type interpolate_mode: str
    :return: resized feature map
    :rtype: torch.Tensor
    """
    in_shape = list(x.shape)
    if not isinstance(out_shape, list):
        out_shape = list(out_shape)
    if len(out_shape) == 3 and len(in_shape) == 4:
        out_shape.insert(0, in_shape[0])
    assert len(in_shape) == len(out_shape) and in_shape[0] == out_shape[0], \
        'Cannot resize tensor from {} to {}'.format(tuple(in_shape), tuple(out_shape))

    # shrink channels
    if in_shape[1] > out_shape[1]:
        x = x[:, :out_shape[1]]

    # shrink spatial axes.
    if len(in_shape) == 4 and (in_shape[2] > out_shape[2] or in_shape[3] > out_shape[3]):
        assert in_shape[2] % out_shape[2] == 0 and in_shape[3] % out_shape[3] == 0
        scale_factor = (in_shape[2] // out_shape[2],
                        in_shape[3] // out_shape[3])
        x = F.avg_pool2d(x,
                         kernel_size=scale_factor,
                         stride=scale_factor,
                         ceil_mode=False,
                         padding=0,
                         count_include_pad=False)

    # extend spatial axes
    if in_shape[2] < out_shape[2]:
        assert out_shape[2] % in_shape[2] == 0 and \
               out_shape[2] / in_shape[2] == out_shape[3] / in_shape[3]
        scale_factor = out_shape[2] // in_shape[2]

        if interpolate_mode == 'bilinear':
            x = F.interpolate(x, scale_factor=scale_factor, mode='bilinear', align_corners=True)
        else:
            x = F.interpolate(x, scale_factor=scale_factor, mode=interpolate_mode)

    # extend channels
    if in_shape[1] < out_shape[1]:
        z = torch.zeros([x.shape[0], out_shape[1] - in_shape[1]] + out_shape[2:]).to(x.device)
        x = torch.cat([x, z], 1)

    return x


def flatten(tensor):
    """
    Flatten input tensor as the shape of (nb, nf)

    :param tensor: input Tensor
    :type tensor: torch.Tensor
    :return:  flattened tensor
    :rtype: torch.Tensor
    """
    assert len(tensor.shape) >= 2

    if len(tensor.shape) > 2:
        flattened = tensor.view(tensor.shape[0], -1)
    else:
        flattened = tensor

    return flattened
