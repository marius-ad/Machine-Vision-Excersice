#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Automatic hysteresis thresholding

Author: Marius Adamske
MatrNr: 12618651
"""

import cv2
import numpy as np

from hyst_thresh import hyst_thresh


def hyst_thresh_auto(edges_in: np.array, low_prop: float, high_prop: float) -> np.array:
    """ Apply automatic hysteresis thresholding.

    Apply automatic hysteresis thresholding by automatically choosing the high
    and low thresholds of standard hysteresis thresholding. low_prop is the
    proportion of nonzero edge pixels remaining after non-maximum suppression
    which should lie above the low threshold, and high_prop is the proportion
    above the high threshold.

    Exclude zero-valued background pixels when calculating the thresholds.
    Small differences due to percentile interpolation or rounding are acceptable.
    Pass the original, spatially arranged input image to hyst_thresh without
    normalizing it again. If there are no edge pixels, return an all-zero image.
    Do not modify the input array.

    :param edges_in: Edge strength of the image in range [0., 1.]
    :type edges_in: np.array with shape (height, width) with dtype = np.float32 and values in the range [0., 1.]

    :param low_prop: Proportion of nonzero edge pixels which should lie above the low threshold
    :type low_prop: float in range [0., 1.]

    :param high_prop: Proportion of nonzero edge pixels which should lie above the high threshold
    :type high_prop: float in range [0., 1.]

    :return: Binary edge image
    :rtype: np.array with shape (height, width) with dtype = np.float32 and values either 0 or 1
    """
    ######################################################
    # Write your own code here
    # Calculate the low and high thresholds based on the proportions

    # Check if there are any non-zero edge pixels  
    if np.count_nonzero(edges_in) == 0:
        return np.zeros_like(edges_in, dtype=np.float32)

    # Calculate low, high thresholds based low_prop, high_prop   
    low_threshold = np.percentile(edges_in[edges_in > 0], (1 - low_prop) * 100)
    high_threshold = np.percentile(edges_in[edges_in > 0], (1 - high_prop) * 100)

    #print(low_threshold, high_threshold)

    # run hysteresis thresholding with the calculated thresholds
    hyst_out = hyst_thresh(edges_in, low_threshold, high_threshold)

    ######################################################
    return hyst_out
