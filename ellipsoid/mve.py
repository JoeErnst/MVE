import numpy
from scipy.stats import chi2

# ------------------------------------------------------------------------------
# MINIMUM VOLUME ELLIPSOID ANALYSIS
# ------------------------------------------------------------------------------


class mve(object):
    """
    Estimates the Minimum Volume Ellipsoid.

    For reading see: 
    http://onlinelibrary.wiley.com/doi/10.1002/env.628/abstract
    http://scikit-learn.org/stable/modules/generated/sklearn.covariance.MinCovDet.html
    
    Parameters
    -----------
    n_samples: non-negative integer
        How many times to sample from the population. Runtime depends linearly on this figure.

    self.required_n_data: non-negative integer
        Number of datapoints included in a minimum volume ellipsoid. If not given it is set to (n_data + n_features + 1)/2

    singularity_add_samples: non-negative integer
        In case a drawn sample yields a singular covariance matrix, how many data points to add

    random_state: non-negative integer
        Random state or seed
    
    Attributes
    -----------
    n_data: non-negative integer
        number of data points in population
    n_features: non-negative integer
        number of features in the dataset

    P_J = float('inf')
    resulting_data: numpy nd array
        Data points included in the final MVE.
    resulting_indices: numpy nd array
        Indices of final data points
    mean_hat: numpy nd array
        Center of the final MVE.
    vcov_hat: numpy nd array
        Covariance matrix of finale MVE.

    References
    -----------
    Jackson, D.A. and Chen, Y., 2004. Robust principal component analysis and outlier detection with ecological data. Environmetrics, 15(2), pp.129-139.

    Van Aelst, S. and Rousseeuw, P., 2009. Minimum volume ellipsoid. Wiley Interdisciplinary Reviews: Computational Statistics, 1(1), pp.71-82.
    """

    def __init__(
        self,
        n_samples = 10000,
        required_n_data = None
        singularity_add_samples = 1, 
        random_state = None
    ):

        self.n_samples = n_samples
        self.required_n_data = None
        self.singularity_add_samples = singularity_add_samples
        self.random_state = random_state

        if random_state is not None:
            numpy.random.seed(random_state)

    def get_params(self):
        """Return parameters as a dictionary.
        # TODO potentially later inherit from EmpiricalCovariance in SciKit Learn

        Returns
        -------
        Dictionary of parameters
        """
        params = {
            'n_samples': self.n_samples, 
            'required_n_data': self.required_n_data,
            'singularity_add_samples': self.singularity_add_samples, 
            'random_state': self.random_state
        }
        return params

    def set_params(self, **parameters):
        """Set parameters
        Parameters
        -----------
        conditional arguments

        Returns
        ----------
        self
        """
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self


    def fit(self, X):
        """Fits a minimum volume ellipsoid given data X
        Parameters
        -----------
        X: numpy nd array (axis 0: datapoints, axis 1: features)

        Returns
        ----------
        (numpy nd array center of ellipsoid, numpy nd array covariance matrix of ellipsoid)
        """
        if not isinstance(X, numpy.ndarray):
            try:
                X = numpy.array(X, dtype='float')
            except:
                raise ValueError("X must be (numpy)-array like.")

        if any([len(datapoint) != len(X[0]) for datapoint in X]):
            raise ValueError("Inhomogenous dataset.")

        tolerance = 0.1
        if numpy.abs(numpy.linalg.det(numpy.cov(X.transpose()))) <= tolerance:
            raise ValueError("Lack of variance in data.", numpy.abs(numpy.linalg.det(numpy.cov(X))))

        self.n_data, self.n_features = X.shape

        if self.required_n_data is None:
            self.required_n_data = int((self.n_data + self.n_features + 1) / 2)
        self.P_J = float('inf')
        self.resulting_data = numpy.array([])
        self.resulting_indices = numpy.array([])

        for i in range(0, self.n_samples):
            # potentially allow for deterministic order as well
            sample_indices = numpy.random.choice(range(0, self.n_data), size=self.n_features+1, replace=False)
            sample_data = X[sample_indices].transpose()

            mean = sample_data.mean(axis=1)
            vcov = numpy.cov(sample_data)

            max_iter_singularity = self.required_n_data
            j = 0
            while numpy.linalg.det(vcov) == 0 and j <= max_iter_singularity:
                # prevent duplicated indices 
                remaining_indices = numpy.setdiff1d(range(0, self.n_data), sample_indices)
                add_sample_index = numpy.random.choice(remaining_indices, size=self.singularity_add_samples)
                # prevent duplicated indices 

                sample_indices = numpy.append(sample_indices, add_sample_index)

                sample_data = X[sample_indices].transpose()
                vcov = numpy.cov(sample_data)

                j = j + 1

            if numpy.linalg.det(vcov) == 0:
                raise ValueError("Singular Data")

            # either for loop or a lot of redundant caluclations (but vectorised)
            X_minus_mean = X - numpy.tile(mean, (self.n_data,1))      
            m_J_squared_array = numpy.diag(X_minus_mean.dot(numpy.linalg.inv(vcov)).dot(X_minus_mean.transpose()))
            m_J_squared = numpy.sort(m_J_squared_array)[self.required_n_data]

            P_J_tmp = numpy.sqrt(m_J_squared ** self.n_features * numpy.linalg.det(vcov))
            if self.P_J > P_J_tmp:
                self.resulting_indices = sample_indices.copy()
                self.resulting_data = X[sample_indices].transpose()
                self.P_J = P_J_tmp

        sample_correction_term = (1 + 15 / (self.n_data - self.n_features)) **2

        chi2_med = chi2.median(self.n_features, loc=0, scale=1)
        
        T_X = self.resulting_data.mean(axis=1)
        C_X = sample_correction_term * (1 / chi2_med) * m_J_squared * numpy.cov(self.resulting_data)
        
        self.mean_hat = T_X
        self.vcov_hat = C_X

        return (self.mean_hat, self.vcov_hat)

    def get_distances(self, X=None):
        '''Given center and covariance matrix of an ellispoid, calculates a ranking for data X

        Parameters
        -----------
        X: numpy nd array (axis 0: datapoints, axis 1: features)
        
        Returns
        --------
        numpy nd array in length of number of datapoints
        '''

        if X is None:
            X = self.X
        
        X_minus_mean = X - numpy.tile(self.mean_hat, (len(X), 1))    

        return numpy.diag(X_minus_mean.dot(numpy.linalg.inv(self.vcov_hat)).dot(X_minus_mean.transpose()))


if __name__ == "__main__":
    pass

