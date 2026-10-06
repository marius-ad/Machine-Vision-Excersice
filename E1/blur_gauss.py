#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Blur the input image with Gaussian filter kernel

Author: Marius Adamske
MatrNr: 12618651
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
    
    # dimensions of the kernel dependend on sigma
    k_w = 2 * math.ceil(3 * sigma) +1
    #k_w = 19 #for testing
    x, y = np.indices((k_w, k_w))

    # gaussian filter
    # calculate values dependend on indices and distance to middle indice 
    middle = (k_w - 1) / 2
    gauss_filter = 1 / (2 * math.pi * (sigma ** 2)) * np.exp(-((x - middle) ** 2 + (y - middle) ** 2) / (2 * (sigma ** 2)))

    # normalize
    gauss_filter /= gauss_filter.sum()

    # check if filter values add up to approx 1
    # print(gaus_filter.sum(axis=0).sum(axis=0))

    img_blur = cv2.filter2D(img, -1, gauss_filter)

    ######################################################
    return img_blur
