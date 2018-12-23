import glob
import os

import cv2
import numpy as np


def get_image_list(image_dir, recursive=True):
    """
    Get list of image paths

    :param image_dir: path to image directory
    :type image_dir: str
    :param recursive: whether or not to use recursive mode
    :type recursive: bool
    :return: list of image paths
    :rtype: list[str]
    """
    image_list = glob.glob(os.path.join(image_dir, '*.png'), recursive=recursive)
    return sorted(image_list)


def read_images(image_list,
                image_num,
                start_idx=0,
                skip=1,
                invalid_list=None,
                use_mp=False):
    """
    Read numbers of images from list

    :param image_list: list of image files
    :type image_list: list[str]
    :param image_num: number of images to read
    :type image_num: int
    :param start_idx: start index of image to read
    :type start_idx: int
    :param skip: display skip
    :type skip: int
    :param invalid_list: list of invalid image (would br converted to gray scale)
    :type invalid_list: list[int]
    :param use_mp: whether or not to read images by multi processes
    :type use_mp: bool
    :return: image objects in the shape of (N, H, W, C)
    :rtype: numpy.ndarray
    """

    def read_wrapper(image_idx):
        image_filename = image_list[image_idx]
        image_temp = cv2.imread(os.path.join(image_filename))
        if invalid_list is not None and image_idx in invalid_list:
            gray_image = cv2.cvtColor(image_temp, cv2.COLOR_BGR2GRAY)
            image_temp = np.stack((gray_image,) * 3, axis=-1)
        return image_temp

    # determine indices
    image_num = min(image_num * skip, len(image_list) - start_idx + 1)
    image_indices = list(range(start_idx,
                               start_idx + image_num,
                               skip))
    # read images
    if use_mp:
        import pathos.multiprocessing as mp
        with mp.Pool(mp.cpu_count() - 1) as p:
            images = p.map(read_wrapper, image_indices)
    else:
        images = list(map(read_wrapper, image_indices))

    # reshape
    images = np.stack(images, axis=0)

    return images


def create_image_grid(images, grid_size=None):
    """
    Create a image grid with N frames by given images

    :param images: some images in the shape of (N, H, W, C)
    :type images: numpy.ndarray
    :param grid_size: grid size like (grid_w, grid_h) or (grid_w, )
    :type grid_size: tuple
    :return: image grid
    :rtype: numpy.ndarray
    """
    # get image shape
    assert images.ndim == 4
    num_img, img_c, img_w, img_h = images.shape

    # determine widht and height of gridh
    if grid_size is not None:
        assert len(grid_size) in (1, 2)
        if len(grid_size) == 2:
            grid_w, grid_h = grid_size
        else:
            grid_w = grid_size[0]
            grid_h = max((num_img - 1) // grid_w + 1, 1)
    else:
        grid_w = max(int(np.ceil(np.sqrt(num_img))), 1)
        grid_h = max((num_img - 1) // grid_w + 1, 1)

    # create image grid
    grid = np.zeros([grid_h * img_h, grid_w * img_w, img_c],
                    dtype=images.dtype)
    for idx in range(num_img):
        x = (idx % grid_w) * img_w
        y = (idx // grid_w) * img_h
        grid[y:y + img_h, x:x + img_w, ...] = images[idx]
    return grid


def create_image_sequence(images, grid_size):
    """
    Create a image grid sequence in (cols, rows) grid with N frames by given images

    :param images: some images in the shape of (N * rows * cols, H, W, C)
    :type images: numpy.ndarray
    :param grid_size: grid size like (cols, rows) or (cols, )
    :type grid_size: tuple
    :return: image grid sequence in the shape of (N, H * rows, W * cols, C)
    :rtype: numpy.ndarray
    """
    # get image shape
    assert images.ndim == 4
    num_images, nc, nw, nh = images.shape

    # determine width and height of grid
    assert len(grid_size) in (1, 2)
    if len(grid_size) == 2:
        grid_w, grid_h = tuple(grid_size)
    else:
        grid_w = grid_size[0]
        grid_h = max((num_images - 1) // grid_w + 1, 1)

    # create sequence of image grids
    num_frames = num_images // (grid_w * grid_h)
    seq = []
    for frame_idx in range(num_frames):
        seq.append(create_image_grid(
            images[frame_idx::grid_w * grid_h], grid_size=(grid_w, grid_h)))
    seq = np.stack(seq, axis=0)
    return seq


def show_image_grid(grid,
                    scale=0.5,
                    window_name='image grid',
                    wait_time=0):
    """
    Show image grid with scale factor

    :param grid: image grid in the shape of (H, W, C)
    :type grid: numpy.ndarray
    :param scale: image scale factor
    :type scale: float
    :return: pressed key
    :param window_name: window name
    :type window_name: str
    :param wait_time: waiting time for keyboard operation
    :type wait_time: int
    :rtype: int
    """
    # resize the image
    if scale != 1.0:
        scale_h = int(scale * grid.shape[0])
        scale_w = int(scale * grid.shape[1])
        grid = cv2.resize(grid, (scale_w, scale_h))

    # show image sequence
    # cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.imshow(window_name, grid)

    # wait for key
    key_code = cv2.waitKey(wait_time)

    return key_code


def show_image_sequence(seq,
                        scale=0.5,
                        window_name='image sequence',
                        wait_time=0):
    """
    Show image grid with scale factor

    :param seq: image seq in the shape of (N, H, W, C), N denotes the number of frames
    :type seq: numpy.ndarray
    :param scale: image scale factor
    :type scale: float
    :param window_name: window name
    :type window_name: str
    :param wait_time: waiting time for keyboard operation
    :type wait_time: int
    :return: pressed key
    :rtype: int
    """
    # resize the image
    if scale != 1.0:
        seq_resized = []
        scale_h = int(scale * seq.shape[1])
        scale_w = int(scale * seq.shape[2])
        for idx in range(seq.shape[0]):
            seq_resized.append(cv2.resize(seq[idx, :], (scale_w, scale_h)))
        seq = np.stack(seq_resized, axis=0)

    # show image sequence
    for idx in range(seq.shape[0]):
        # cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.imshow(window_name, seq[idx, :])
        cv2.waitKey(30)

    # wait for key
    key_code = cv2.waitKey(wait_time)

    return key_code


def plot_text(image,
              text,
              position,
              offset=(0, 0),
              color=None,
              font_scale=1,
              thickness=3):
    """
    Plot text on given image

    :param image: image
    :type image: numpy.ndarray
    :param text: text to be drawn
    :type text: str
    :param position: text position (x, y)
    :type position: tuple(int)
    :param offset: text position offset (x, y)
    :type offset: tuple(int)
    :param color: text color (r, g, b)
    :type color: tuple(int)
    :param font_scale: scale of font size
    :type font_scale: float or int
    :param thickness: thickness of font
    :type thickness: float or int
    :return: processed image
    :rtype: numpy.ndarray
    """
    if color is None:
        color = (0, 0, 0)
    cv2.putText(img=image,
                text=text,
                org=(position[0] + offset[0], position[1] + offset[1]),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=font_scale,
                color=color,
                thickness=thickness)


def plot_text_on_sequence(image_seq,
                          text,
                          position,
                          color=None,
                          font_scale=1,
                          thickness=3):
    """
    Plot text on given image sequence

    :param image_seq: image
    :type image_seq: numpy.ndarray
    :param text: text to be drawn
    :type text: str
    :param position: text position (x, y)
    :type position: tuple(int)
    :param color: text color (r, g, b)
    :type color: tuple(int)
    :param font_scale: scale of font size
    :type font_scale: float or int
    :param thickness: thickness of font
    :type thickness: float or int
    :return: processed image
    :rtype: numpy.ndarray
    """
    for image_idx in range(len(image_seq)):
        plot_text(image_seq[image_idx],
                  text=text,
                  color=color,
                  position=position,
                  font_scale=font_scale,
                  thickness=thickness)


def plot_text_on_grid(image_grid, grid_size, text, position, color=None):
    """
    Plot text on given image grid

    :param image_grid: image grid in the shape of (H, W, C)
    :type image_grid: numpy.ndarray
    :param grid_size: grid size like (cols, rows)
    :type grid_size: tuple
    :param text: text to be drawn
    :type text: str or list
    :param position: text position (x, y)
    :type position: tuple(int)
    :param color: text color (r, g, b)
    :type color: tuple(int) or list[tuple(int)]
    :return: processed image
    :rtype: numpy.ndarray
    """
    # get grid size
    assert len(grid_size) == 2
    grid_w, grid_h = grid_size
    num_images = grid_w * grid_h

    # get image shape
    assert len(grid_size) == 2
    img_h, img_w, _ = image_grid.shape

    # pre-process text and color
    if isinstance(text, str):
        text = [text] * num_images
    if isinstance(color, tuple):
        color = [color] * num_images
    assert len(text) == num_images and len(color) == num_images

    # plot text
    offset_x = int(img_w / grid_w)
    offset_y = int(img_h / grid_h)
    for image_idx in range(num_images):
        plot_text(image_grid,
                  text=text[image_idx],
                  color=color[image_idx],
                  position=position,
                  offset=(image_idx // grid_w * offset_x,
                          image_idx // grid_h * offset_y))
