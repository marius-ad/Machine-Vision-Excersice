#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Blur the input image with Gaussian filter kernel

Author: FILL IN
MatrNr: FILL IN
"""

import cv2
import math
import numpy as np

def blur_gauss(img: np.array, sigma: float) -> np.array:
    """ Blur the input image with a Gaussian filter with standard deviation of sigma.

    Construct a two-dimensional Gaussian kernel with standard deviation sigma and
    size 2 * c(3 * sigma) + 1 in each dimenstion. Normalize the kernel so its
    values sum to one, then apply it to the input image using cv2.filter2D.
    Do not modify the input image.

    :param img: Grayscale input image
    :type img: np.array with shape (height, width) with dtype = np.float32 and values in the range [0., 1.]

    :param sigma: The standard deviation of the Gaussian kernel
    :type sigma: float

    :return: Blurred image
    :rtype: np.array with shape (height, width) with dtype = np.float32 and values in the range [0.,1.]
    """
    ######################################################
    # Write your own code here
    
    kernel_width = 2 * math.ceil(3 * sigma) +1
    print(kernel_width)

    square_filter = np.zeros((kernel_width, kernel_width))
    square_filter[:] = 1 / (kernel_width ** 2)

    print(square_filter.sum(axis=0).sum(axis=0))

    img_blur = cv2.filter2D(img, -1, square_filter)

    ######################################################
    return img_blur
