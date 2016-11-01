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
    
    TODO
    Multiple ways of handling Singularity issues of covariance matrix (jump, add epsilon, add data)
    more flexible data handling
    Readme file
    add nice examples
    """


    def __init__(
        self,
        n_samples = 10000
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
        # TODO add test if enough variance is existant

        self.n_features = len(X[0])
        self.n_data = len(X)
        self.required_n_data = int((self.n_data + self.n_features + 1) / 2)
        self.P_J = float('inf')
        self.resulting_data = np.array([])
        self.resulting_indices = np.array([])

        for i in range(0, self.n_samples):
            # TODO allow for deterministic order as well
            sample_indices = np.random.choice(range(0, self.n_data), size=self.n_features+1, replace=False)
            sample_data = X[sample_indices].transpose()

            mean = sample_data.mean(axis=1)
            vcov = np.cov(sample_data)

            # TODO add to parameters
            # Add data if VCOV Matrix is singular
            max_iter_singularity = self.required_n_data
            j = 0
            while np.linalg.det(vcov) == 0 and j <= max_iter_singularity:
                add_sample_index = np.random.choice(range(0, self.n_data), size=1)
                # prevent duplicated indices 
                # TODO make more efficient, e.g. with set operations
                if any(sample_indices == add_sample_index):
                    continue
                sample_indices = np.append(sample_indices, add_sample_index)

                sample_data = X[sample_indices].transpose()
                vcov = np.cov(sample_data)

                j = j + 1

            if np.linalg.det(vcov) == 0:
                raise ValueError("Singular Data")

            # either for loop or a lot of redundant caluclations (but vectorised)
            X_minus_mean = X - np.tile(mean, (self.n_data,1))      
            m_J_squared_array = np.diag(X_minus_mean.dot(np.linalg.inv(vcov)).dot(X_minus_mean.transpose()))
            m_J_squared = np.sort(m_J_squared_array)[self.required_n_data]

            P_J_tmp = np.sqrt(m_J_squared ** self.n_features * np.linalg.det(vcov))
            if self.P_J > P_J_tmp:
                self.resulting_indices = sample_indices.copy()
                self.resulting_data = X[sample_indices].transpose()
                self.P_J = P_J_tmp

        sample_correction_term = (1 + 15 / (self.n_data - self.n_features)) **2

        chi2_med = st.chi2.median(self.n_features, loc=0, scale=1)
        
        T_X = self.resulting_data.mean(axis=1)
        C_X = sample_correction_term * (1 / chi2_med) * m_J_squared * np.cov(self.resulting_data)
        
        self.mean_hat = T_X
        self.vcov_hat = C_X
        self.X = X.copy()

        return 1

    def get_distances(self, X=None):
        if X is None:
            X = self.X
        
        X_minus_mean = X - np.tile(self.mean_hat, (len(X), 1))    

        return np.diag(X_minus_mean.dot(np.linalg.inv(self.vcov_hat)).dot(X_minus_mean.transpose()))


if __name__ == "__main__":
    mymve = mve()

    mymve.set_params(n_samples=5000)

    from sklearn.datasets import load_boston
    X1 = load_boston()['data'][:, [8, 10]]  # two clusters
    # X1 = load_boston()['data'][:, [5, 12]]  # "banana"-shaped

    mymve.fit(X1)

    distances = mymve.get_distances()


    # ANALYSE RESULTS GRAPHICALLY
    percentile = 0.97

    binary_distances = distances <= np.percentile(distances, percentile * 100, axis=0)

    import matplotlib.pyplot as plt

    plt.scatter(X1.transpose()[0], X1.transpose()[1], c=distances)
    
    for i, txt in enumerate(distances):
        if txt >= np.percentile(distances, percentile * 100, axis=0):
            plt.annotate(txt, (X1.transpose()[0][i],X1.transpose()[1][i]))

    #plt.colorbar()
    #plt.scatter(X1.transpose()[0, binary_distances], X1.transpose()[1, binary_distances])
    plt.show()

