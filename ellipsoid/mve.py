"""Minimum Volume Ellipsoid classes."""

import numpy as np
from scipy import stats as st

# ------------------------------------------------------------------------------
# MINIMUM VOLUME ELLIPSOID ANALYSIS
# ------------------------------------------------------------------------------


class mve(object):

    """
    Base Minimum Volume Ellipsoid Analysis class.
    PAPER: 
    http://onlinelibrary.wiley.com/doi/10.1002/env.628/abstract
    http://scikit-learn.org/stable/modules/generated/sklearn.covariance.MinCovDet.html
    
    TICKETS
    finish implementation
    Singularity issues of covariance matrix (jump, add epsilon, add data)
    """

    def __init__(
        self,
        n_samples=10000
    ):
        self.n_samples = n_samples


    def get_params(self):
        """Return parameters as a dictionary.
        # TODO potentially later inherit from EmpiricalCovariance in SciKit Learn
        """
        params = {
            'n_samples': self.n_samples
        }
        return params

    def set_params(self, **parameters):
        """Set parameters using kwargs."""
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

    def fit(self, X):
        # TODO insert test of missing data
        # TODO accept multiple datatypes
        self.n_features = len(X[0])
        self.n_data = len(X)

        for i in range(0, self.n_samples):
            print i


if __name__ == "__main__":
    mymve = mve()

    mymve.set_params(n_samples=2000)
    print mymve.get_params()

    from sklearn.datasets import load_boston
    X1 = load_boston()['data'][:, [8, 10]]  # two clusters

    mymve.fit(X1)



