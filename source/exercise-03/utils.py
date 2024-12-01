# Description:
#   Exercise3 utils.py.
#
# Copyright (C):
# v1: 2018 Santiago Cortes, Juha Ylioinas
# v2: 2025 Iaroslav Melekhov
#
# This software is distributed under the GNU General Public
# Licence (version 2 or later); please refer to the file
# Licence.txt, included with the software, for details.

from __future__ import division

from enum import Enum
from types import *

import numpy as np
from PIL import Image
from scipy.ndimage.interpolation import map_coordinates


# Define an Enum for noise types
class NoiseType(Enum):
    SALT_AND_PEPPER = "salt_and_pepper"
    GAUSSIAN = "gaussian"


def rgb_to_grayscale(image_path: str, size: tuple[int, int] = None) -> np.ndarray:
    """
    Convert an RGB image to grayscale, resize it, and return it as a NumPy array.

    Args:
        image_path (str): Path to the input RGB image.
        size (tuple, optional): Target size as (width, height). If None, no resizing is performed.

    Returns:
        numpy.ndarray: Grayscale image with shape (H, W) and values in the range [0, 1].
    """
    # Open the image and convert it to RGB
    image = Image.open(image_path)

    # Resize the image if a size is provided
    if size:
        image = image.resize(size, Image.LANCZOS)

    # Convert the image to grayscale
    grayscale_image = image.convert("L")

    # Convert the grayscale image to a NumPy array and scale values to [0, 1]
    grayscale_array = np.asarray(grayscale_image, dtype=np.float64) / 255.0

    return grayscale_array


def add_noise(image: np.ndarray, noise_type: NoiseType, **kwargs) -> np.ndarray:
    """
    Add noise to an image using the specified noise type.

    Args:
        image (np.ndarray): Input image, a 2D (grayscale) or 3D (color) NumPy array with values in the range [0, 1].
        noise_type (NoiseType): Type of noise to apply (SALT_AND_PEPPER or GAUSSIAN).
        **kwargs: Additional arguments specific to the noise type.
            For "salt_and_pepper", provide 'prob' (float).
            For "gaussian", provide 'std_dev' (float).

    Returns:
        np.ndarray: Noisy image with the same shape as the input.
    """
    if noise_type == NoiseType.SALT_AND_PEPPER:
        prob = kwargs.get("prob")
        if prob is None:
            print(f"The 'prob' argument is empty, using default value 0.05")
            prob = 0.05
        return _add_salt_and_pepper_noise(image, prob)

    elif noise_type == NoiseType.GAUSSIAN:
        std_dev = kwargs.get("std_dev")
        if std_dev is None:
            print(f"The 'std_dev' argument is empty, using default value 0.05")
            std_dev = 0.05
        return _add_gaussian_noise(image, std_dev)

    else:
        raise ValueError(f"Unsupported noise type: {noise_type}")


def _add_salt_and_pepper_noise(image: np.ndarray, prob: float) -> np.ndarray:
    """
    Add Salt and Pepper noise to an input image using a single probability.

    Args:
        image (np.ndarray): Input image, a 2D (grayscale) or 3D (color) NumPy array with values in the range [0, 1].
        prob (float): Probability for each pixel to be altered with Salt (1) or Pepper (0).

    Returns:
        np.ndarray: Noisy image with the same shape as the input.
    """
    if not (0 <= prob <= 1):
        raise ValueError("Probability must be in the range [0, 1].")

    # Copy the input image to avoid modifying the original
    noisy_image = np.copy(image)

    # Generate random matrix
    random_matrix = np.random.random(image.shape)

    # Apply Salt and Pepper noise
    noisy_image[random_matrix < prob] = 0.0  # Pepper: set to black (min intensity)
    noisy_image[random_matrix >= 1 - prob] = 1.0  # Salt: set to white (max intensity)

    return noisy_image


def _add_gaussian_noise(image: np.ndarray, std_dev: float) -> np.ndarray:
    """
    Add zero-mean Gaussian noise to an input image.

    Args:
        image (np.ndarray): Input image, a 2D (grayscale) or 3D (color) NumPy array with values in the range [0, 1].
        std_dev (float): Standard deviation of the Gaussian noise.

    Returns:
        np.ndarray: Noisy image with the same shape as the input, values clipped to [0, 1].
    """
    if std_dev < 0:
        raise ValueError("Standard deviation must be non-negative.")

    # Generate Gaussian noise
    noise = np.random.normal(0, std_dev, image.shape)

    # Add the noise to the original image
    noisy_image = image + noise

    # Clip the resulting image to ensure values are in the range [0, 1]
    noisy_image = np.clip(noisy_image, 0, 1)

    return noisy_image


# convert from rgb to grayscale image
def rgb2gray(rgb):

    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    gray = 0.2125 * r + 0.7154 * g + 0.0721 * b

    return gray


# salt-and-pepper noise generator
def imnoise(img, mode, prob):
    imgn = img.copy()
    if mode == "salt & pepper":
        assert prob >= 0 and prob <= 1, "prob must be a scalar between 0 and 1"
        h, w = imgn.shape
        prob_sp = np.random.rand(h, w)
        imgn[prob_sp < prob] = 0
        imgn[prob_sp > 1 - prob] = 1

    return imgn


# Gaussian noise generator
def add_gaussian_noise(img, noise_sigma):
    temp_img = np.copy(img)
    h, w = temp_img.shape
    noise = np.random.randn(h, w) * noise_sigma
    noisy_img = temp_img + noise
    return noisy_img


# 2d Gaussian filter
def gaussian2(sigma, N=None):

    if N is None:
        N = 2 * np.maximum(4, np.ceil(6 * sigma)) + 1

    k = (N - 1) / 2.0

    xv, yv = np.meshgrid(np.arange(-k, k + 1), np.arange(-k, k + 1))

    # 2D gaussian filter
    g = 1 / (2 * np.pi * sigma**2) * np.exp(-(xv**2 + yv**2) / (2 * sigma**2))

    # 1st order derivatives
    gx = -xv / (2 * np.pi * sigma**4) * np.exp(-(xv**2 + yv**2) / (2 * sigma**2))
    gy = -yv / (2 * np.pi * sigma**4) * np.exp(-(xv**2 + yv**2) / (2 * sigma**2))

    # 2nd order derivatives
    gxx = (-1 + xv**2 / sigma**2) * np.exp(-(xv**2 + yv**2) / (2 * sigma**2)) / (2 * np.pi * sigma**4)
    gyy = (-1 + yv**2 / sigma**2) * np.exp(-(xv**2 + yv**2) / (2 * sigma**2)) / (2 * np.pi * sigma**4)
    gxy = (xv * yv) / (2 * np.pi * sigma**6) * np.exp(-(xv**2 + yv**2) / (2 * sigma**2))

    return g, gx, gy, gxx, gyy, gxy


# fit an affine model between two 2d point sets
def affinefit(x, y):
    # Ordinary least squares (check wikipedia for further details):
    #
    # Y                          = P*X_aug            % X_aug is in homogenous coords (one sample per col)
    # Y'                         = X_aug'*P'          % take transpose from both sides
    # X_aug*Y'                   = X_aug*X_aug'*P'    % multiply both sides from left by X_aug
    # inv(X_aug*X_aug')*X_aug*Y' = P'                 % multiply both sides from left by the inverse of X_aug*X_aug'
    n = x.shape[0]
    x = x.T
    y = y.T
    x_aug = np.concatenate((x, np.ones((1, n))), axis=0)
    y_aug = np.concatenate((y, np.ones((1, n))), axis=0)
    xtx = np.dot(x_aug, x_aug.T)
    xtx_inv = np.linalg.inv(xtx)
    xtx_inv_x = np.dot(xtx_inv, x_aug)
    P = np.dot(xtx_inv_x, y_aug.T)
    A = P.T[0:2, 0:2]
    b = P.T[0:2, 2]

    return A, b
