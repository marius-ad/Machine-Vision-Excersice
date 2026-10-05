#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Hysteresis thresholding

Author: Marius Adamske
MatrNr: 12618651
"""

import cv2
import numpy as np


def hyst_thresh(edges_in: np.array, low: float, high: float) -> np.array:
    """ Apply hysteresis thresholding.

      Apply hysteresis thresholding to return the edges as a binary image. All
      eight-connected pixels with value > low are considered a valid edge if at
      least one pixel has a value > high.

      Perform threshold comparisons on the floating-point input values without
      normalizing them again. For cv2.connectedComponents, use an np.uint8 mask
      with zero-valued background and nonzero foreground pixels.

      If no pixels have a value > high, return an all-zero image.
      Do not modify the input array.

    :param edges_in: Edge strength of the image in range [0.,1.]
    :type edges_in: np.array with shape (height, width) with dtype = np.float32 and values in the range [0., 1.]

    :param low: Threshold which a pixel's value must exceed to belong to an edge
    :type low: float in range [0., 1.]

    :param high: Threshold which at least one pixel in a connected edge must exceed
    :type high: float in range [0., 1.]

    :return: Binary edge image
    :rtype: np.array with shape (height, width) with dtype = np.float32 and values either 0 or 1
    """
    ######################################################
  
    # compute binary image of pixels above low/high threshold
    edges_low = np.where(edges_in > low, 1, 0).astype(np.uint8)
    edges_high = edges_in > high

    # compute connected components of edges_low, 
    # output of cv2.connectedComponents is a tuple (num_labels, labels)
    num_labels, labels = cv2.connectedComponents(edges_low, connectivity=8)

    # find connected components that contain at least one pixel above high threshold
    labels_high = np.where(labels * edges_high > 0, labels, 0)

    # create binary image of edges that are connected to a pixel above high threshold
    bitwise_img = np.isin(labels, labels_high[labels_high > 0]).astype(np.float32)

    ######################################################
    return bitwise_img
