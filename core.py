from PIL import Image
import numpy as np
import tensorflow as tf
import vggnet


def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.2989, 0.5870, 0.1140])


def gray2rgb(gray):
    width, height = gray.shape
    rgb = np.empty((width, height, 3), dtype=np.float32)
    rgb[:, :, 0] = rgb[:, :, 1] = rgb[:, :, 2] = gray

    return rgb
