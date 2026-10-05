#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Edge detection with the Sobel filter

Author: Marius Adamske
MatrNr: 12618651
"""

import cv2
import numpy as np


def sobel(img: np.array) -> (np.array, np.array):
    """ Apply the Sobel filter to the input image and return the gradient and the orientation.

    Normalize the gradient magnitudes by dividing by their maximum. If the maximum
    is zero, return an all-zero gradient array. Do not modify the input image.

    :param img: Grayscale input image
    :type img: np.array with shape (height, width) with dtype = np.float32 and values in the range [0., 1.]
    :return: (gradient, orientation): gradient: the normalized edge strength of the image in range [0.,1.],
                                      orientation: angle of gradient in range [-np.pi, np.pi]
    :rtype: Two np.arrays with shape (height, width) and dtype = np.float32
    """
    ######################################################
    
    # check img dtype and range
    img = img.astype(np.float32)
    low, high = np.min(img), np.max(img)
    if low < 0.0 or high > 1.0:
        img = (img - low) / (high - low) if high > low else np.zeros_like(img)

    # define filter kernel
    g_x = np.array([[-1,0,1],
                    [-2,0,2],
                    [-1,0,1]])

    g_y = np.array([[-1,-2,-1],
                    [0,0,0],
                    [1,2,1]])

    # horizontal edge
    x = cv2.filter2D(img, -1, g_x)
    # vertical edge
    y = cv2.filter2D(img, -1, g_y)

    # edge strength
    gradient = np.sqrt(x**2 + y**2)

    # angele of gradient
    # arctan(y/x) only gives angle [-pi/2, pi/2]
    # arctan2 shortcut for:
    #   x > 0:          arctan(y / x)
    #   x < 0, y >= 0:  arctan(y / x) + pi
    #   x < 0, y < 0:   arctan(y / x) - pi
    #   x = 0, y > 0:   pi / 2
    #   x = 0, y < 0:  -pi / 2
    orientation = np.arctan2(y, x)

    # normalize
    if np.max(gradient) == 0:
        gradient = np.zeros_like(gradient)
    else:
        gradient /= np.max(gradient)

    ######################################################
    return gradient, orientation
