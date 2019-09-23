#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Classes for scaling features
"""

from sklearn.utils import check_array
import numpy as np
from scipy import sparse
import torch
from torch.autograd import Variable

class MinMaxScalerTorch:
    """Transform features by scaling each feature to a given range.
       The transformation is given by:
       x_min = x.min(axis=0), x_max = x.max(axis=0)
       x_std = (x-x_min) / (x_max - x_min)
       x_scale = x_std * (max-min) + min


       Parameters:
       ------
       feature_range: tuple (min, max), default=(0, 1)
            excepted range of transformed data.
       ------

       Attributes:
       ------
       min_: ndarray, shape (n_features, )
           desired minimum.

       scale_: ndarray, shape (n_features, )
           desired range, (max-min)

       data_min_: ndarray, shape (n_features, )
            per feature minimun seen in the data

       dat_max_: ndarray, shape (n_features, )
            per feature maximum seen in the data

       Examples:
       ------
    """
    def __init__(self, feature_range=(0, 1)):
        """
        """
        self.feature_range = feature_range
        if feature_range[0] >= feature_range[1]:
            raise ValueError("Minimun of excepted feature range must be" + \
             "smaller than maximum. But Got {}".format(feature_range))

    def _reset(self):
        """reset the state of the scaler
        """
        if hasattr(self, 'scale_'):
            del self.scale_
            del self.min_
            del self.data_min_
            del self.data_max_
            del self.data_range_
            del self.n_sample_seen_

    def transform(self, X):
        """scaling features of X according to feature_range.

           Parameters:
           X, array like, shape[n_samples, n_featuers]
        """
        if not hasattr(self, 'scale_'):
            raise AttributeError("The estimator does not have the attribute: scale")

        X = check_array(X, copy=True, warn_on_dtype=True, estimator=self,\
                dtype=np.float32, force_all_finite="allow-nan")
        X = utils.handle_maxmin_in_range(X, self.data_min_, self.data_max_)

        X *= self.scale_
        X += self.min_
        return X

    def inverse_transform(self, X):
        """ the inverse operation of scalin
        """
        if not hasattr(self, 'scale_'):
            raise AttributeError("The estimator does not have the attribute: scale")

        X = check_array(X, copy=True, warn_on_dtype=True, estimator=self, \
                dtype=np.float32, force_all_finite="allow-nan")
        X -= self.min_
        X /= self.scale_
        return X


class StandardScalerTorch:
    """ Standardize features by removing the mean and scaling to unit variance.
        X = (X - X_mean) / X_var

        Parameters:

        Attributes:
          scale_: ndarray or None, shape(n_features,), handle_zeros_in_scale
           on var_

          mean_: ndarray, shape(n_features,), the mean value of each feature
              in the training dataset.

          var_: ndarray, shape(n_features,)
          The variance of each feature, used to computer scale_

          n_batch_seen: int or array, the number of batch data processed by
            the estimator.

    """
    def __init__(self):
        pass 
    def _reset(self):
        """
        """
        if hasattr(self, 'scale_'):
            del self.scale_
            del self.n_batch_seen_
            del self.mean_

    def transform(self, X):
        if not hasattr(self, 'scale_'):
            raise AttributeError("The estimator does not have the attribute: scale")

        #X = check_array(X, accept_sparse='csr', copy=True, \
        # warn_on_dtype=True, estimator=self, dtype=np.float32, \
        # force_all_finite='allow-nan')
        #if sparse.issparse(X):
        #    raise ValueError("Currrent Version does not support Sparse Matrics")

        X = X - self.mean_
        X = torch.div(X, self.scale_)
        return X

    def inverse_transform(self, X):
        #X = np.asarray(X)
        #X = X.copy()
        X = X * self.scale_
        X = X + self.mean_
        return X

    def load_weight(self, scaler, use_cuda):
        self.scale_ = Variable(torch.from_numpy(scaler.scale_).float())
        self.n_batch_seen_ = scaler.n_batch_seen_
        self.mean_ = Variable(torch.from_numpy(scaler.mean_).float())
        self.var_ = Variable(torch.from_numpy(scaler.var_).float())
        self.scale_ = self.scale_.cuda()
        self.mean_ = self.mean_.cuda()
        self.var_ = self.var_.cuda()
