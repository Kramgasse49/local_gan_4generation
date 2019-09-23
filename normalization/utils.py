#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Util function to format a numeric matrix / vector
(1) handle_zeros_in_scale, replace zero with one for further division operation
(2) handle_maxmin_in_range,
"""
from __future__ import division

import numpy as np

def handle_zeros_in_scale(scale):
    """Make sure that whenever scale is zero, it should be handleed correctly,
      especially as the bottom in division.
    """
    if np.isscalar(scale):
        if scale == 0.:
            scale = 1.
    elif isinstance(scale, np.ndarray):
        scale = scale.copy()
    else:
        raise TypeError("The type of the input should be scalar or np.ndarray")
    return scale

def handle_maxmin_in_range(scale, min_scale=None, max_scale=None):
    """Make sure the scale in the range [min_scale, max_scale],
      Values out of range must be truncated into the pre-defined range.

      Parameters:
         scale, a scalar or numpy.ndarray,  shape [n_sample, n_features]
         min_scale, a scalar or numpy.ndarray, shape[n_features,]
         max_scale, a scalar or numpy.ndarray, shape[n_features,]

      examples:

    """
    max_sub_min = max_scale - min_scale 
    max_sub_min[max_sub_min > 0.] = 0. 
    if np.sum(max_sub_min) < 0.:
        raise ValueError("The min_scale should be smaller than the max_scale")

    if np.isscalar(scale):
        if min_scale is not None:
            scale = max(scale, min_scale)
        if max_scale is not None:
            scale = min(scale, max_scale)
    elif isinstance(scale, np.ndarray):
        scale = scale.copy()
        for i in range(scale.shape[0]):
            if min_scale is not None:
                scale[i] = np.maximum(scale[i], min_scale)
            if max_scale is not None:
                scale[i] = np.minimum(scale[i], max_scale)
        # end for
    else:
        raise TypeError("The type of the input should be scalar or np.ndarray")

    return scale
