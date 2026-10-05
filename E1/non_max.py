#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Non-Maxima Suppression

Author: Marius Adamske
MatrNr: 12618651
"""

import cv2
import numpy as np

def non_max(gradients: np.array, orientations: np.array) -> np.array:
    """ Apply Non-Maxima Suppression and return an edge image.

    Filter out all the values of the gradients array which are not local maxima.
    The orientations are used to check for larger pixel values in the direction of orientation.
    Keep the original magnitude of each surviving pixel and set suppressed pixels to zero.
    Do not normalize the result again or modify either input array.

    :param gradients: Edge strength of the image in range [0.,1.]
    :type gradients: np.array with shape (height, width) with dtype = np.float32 and values in the range [0., 1.]

    :param orientations: angle of gradient in range [-np.pi, np.pi]
    :type orientations: np.array with shape (height, width) with dtype = np.float32 and values in the range [-pi, pi]

    :return: Non-Maxima suppressed gradients
    :rtype: np.array with shape (height, width) with dtype = np.float32 and values in the range [0., 1.]
    """
    ######################################################
    rows, columns = gradients.shape
    edges = gradients.copy()

    # orientation with discrete levels
    # dividing the angle cricle into areas
    # Horizontal:    [-pi, -7pi/8] or [-pi/8, pi/8] or [7pi/8, pi]
    # Vertical:      [-5pi/8, -3pi/8] or [3pi/8, 5pi/8]
    # Diagonal \:    (-7pi/8, -5pi/8) or (pi/8, 3pi/8)
    # Diagonal /:    (-3pi/8, -pi/8) or (5pi/8, 7pi/8)
    for i in range(rows):
        for j in range(columns):
            mag = gradients[i, j]
            angle = orientations[i, j]
            abs_angle = np.abs(angle)

            # Horizontal gradient
            if abs_angle <= np.pi / 8 or abs_angle >= 7 * np.pi / 8:
                neighbors = ((i, j - 1), (i, j + 1))

            # Vertical gradient
            elif 3 * np.pi / 8 <= abs_angle <= 5 * np.pi / 8:
                neighbors = ((i - 1, j), (i + 1, j))

            # Diagonal gradient: upper-left and lower-right
            elif (np.pi / 8 < angle < 3 * np.pi / 8 or
                -7 * np.pi / 8 < angle < -5 * np.pi / 8):
                neighbors = ((i - 1, j - 1), (i + 1, j + 1))

            # Diagonal gradient: lower-left and upper-right
            else:
                neighbors = ((i + 1, j - 1), (i - 1, j + 1))

            (i1, j1), (i2, j2) = neighbors

            # the first neighbor wins a tie.
            if ((0 <= i1 < rows and 0 <= j1 < columns and mag <= gradients[i1, j1]) or
                (0 <= i2 < rows and 0 <= j2 < columns and mag < gradients[i2, j2])):
                edges[i, j] = 0

    ######################################################

    return edges
