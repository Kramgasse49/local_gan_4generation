#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Math functions of speical algorithms
"""
from __future__ import division

import numpy as np

def incremental_mean_and_var(X, last_mean, last_variance, last_batch_count):
    """Calculate mean update and a Youngs and Cramer variance update

        Parameters:
            X: array_like, shape(n_samples, n_features)
             a min batch data for variance update

            last_mean,

            last_variance,

            last_batch_count,

        Returns:
            updated_mean :
            updated_variance:
            updated_batch_count :

        Notes:
        Nans are ignored during the algorithm
    """
    last_sum = last_mean * last_batch_count
    new_sum = np.nanmean(X, axis=0)

    updated_batch_count = last_batch_count + 1

    updated_mean = (last_sum + new_sum) / updated_batch_count

    new_unnormalized_variance = np.nanvar(X, axis=0)
    last_unnormalized_variance = last_variance * last_batch_count
    with np.errstate(divide='ignore', invalid='ignore'):
        last_over_new_count = last_batch_count
        updated_unnormalized_variance = (last_unnormalized_variance  + \
        new_unnormalized_variance + last_over_new_count / updated_batch_count * \
         (last_sum / last_over_new_count - new_sum) ** 2)
    zeros = last_batch_count == 0
    updated_unnormalized_variance[zeros] = new_unnormalized_variance[zeros]
    updated_variance = updated_unnormalized_variance / updated_batch_count

    return updated_mean, updated_variance, updated_batch_count
