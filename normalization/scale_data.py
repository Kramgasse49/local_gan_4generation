#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Classes for scaling features
"""

from sklearn.utils import check_array
import numpy as np
from scipy import sparse

import utils
import extendmath

class Scaler:
    """ Basic class for scaling object

    """
    def __init__(self):
        """
        """
        pass

    def _reset(self):
        """
        """
        raise NotImplementedError("")

    def partial_fit(self, X):
        """ batch fitting for estimating parameters of the estimator
        """
        raise NotImplementedError("")

    def fit(self, X):
        """ Compute the minmum and maximum to be used for later scaling

            Parameters
            ----------
            X, array like, shape [n_samples, n_features], or one Generator
                The data used to compute the per-feature minimum and maximum
                for further scaling along the features axis.
        """
        self._reset()

        if isinstance(X, (list, np.ndarray)):
            self.partial_fit(X)
        else:
            for batch_x in X:
                self.partial_fit(batch_x)
               # add a progress monitor
        #
        return self

    def transform(self, X):
        """ Scaling operation on X according to the estimator
        """
        raise NotImplementedError("")

    def inverse_transform(self, X):
        """ the inverse procedure of transformation
        """
        raise NotImplementedError("")

    def get_parameters_name(self):
        return sort(self.__dict__.keys())

    def save(self, outpath):
        paramters_dict = {key: getattr(self, key) for key in self.get_parameters_name()}
        with open(outpath, 'wb') as fout:
            pickle.dump(paramters_dict, fout)
    def load(self, inpath):
        paramters_dict = pickle.load( open(inpath, 'rb'))
        for name in self.get_parameters_name():
            setattr(self, name, paramters_dict[name])
    #
class MinMaxScaler(Scaler):
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
        #super(MinMaxScaler, self).__init__()
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

    def partial_fit(self, X):
        """Online compuation of min and max on X for further scalin

            Parametere:
            X, array like, shape [n_sample, n_featuers]
        """
        if sparse.issparse(X):
            raise TypeError("MinMaxScaler does not support sparse input.")

        X = check_array(X, copy=True, warn_on_dtype=True, estimator=self, \
                dtype=np.float32, force_all_finite="allow-nan")

        data_min = np.nanmin(X, axis=0)
        data_max = np.nanmax(X, axis=0)
        if not hasattr(self, 'n_sample_seen_'):
            self.n_sample_seen_ = X.shape[0]
        else:
            data_min = np.minimum(self.data_min_, data_min)
            data_max = np.maximum(self.data_max_, data_max)
            self.n_sample_seen_ += X.shape[0]

        data_range = data_max - data_min
        self.scale_ = ((self.feature_range[1] - self.feature_range[0]) / \
                utils.handle_zeros_in_scale(data_range))
        self.min_ = self.feature_range[0] - data_min * self.scale_
        self.data_min_ = data_min
        self.data_max_ = data_max
        self.data_range_ = data_range


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


class StandardScaler(Scaler):
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
    def _reset(self):
        """
        """
        if hasattr(self, 'scale_'):
            del self.scale_
            del self.n_batch_seen_
            del self.mean_
            del self.var_

    def partial_fit(self, X):
        X = check_array(X, accept_sparse=('csr', 'csc'), copy=True, \
          warn_on_dtype=True, estimator=self, dtype=np.float32, \
          force_all_finite='allow-nan')
        if hasattr(self, 'n_batch_seen_') and \
                isinstance(self.n_batch_seen_, (int, np.integer)):
            self.n_batch_seen_ = np.repeat(self.n_batch_seen_, X.shape[1]).astype(np.int64)
        #
        if sparse.issparse(X):
            raise ValueError("Currrent Version does not support Sparse Matrics")

        if not hasattr(self, 'n_batch_seen_'):
            self.n_batch_seen_ = np.zeros(X.shape[1], dtype=np.int64)
        if not hasattr(self, 'scale_'):
            self.mean_, self.var_ = (0., 0.)

        self.mean_, self.var_, self.n_batch_seen_ = \
           extendmath.incremental_mean_and_var(X, self.mean_, self.var_, \
                self.n_batch_seen_)

        self.scale_ = utils.handle_zeros_in_scale(np.sqrt(self.var_))
        #
    def transform(self, X):
        if not hasattr(self, 'scale_'):
            raise AttributeError("The estimator does not have the attribute: scale")

        X = check_array(X, accept_sparse='csr', copy=True, \
         warn_on_dtype=True, estimator=self, dtype=np.float32, \
         force_all_finite='allow-nan')
        if sparse.issparse(X):
            raise ValueError("Currrent Version does not support Sparse Matrics")

        X -= self.mean_
        X /= self.scale_
        return X

    def inverse_transform(self, X):
        X = np.asarray(X)
        X = X.copy()
        X *= self.scale_
        X += self.mean_
        return X
