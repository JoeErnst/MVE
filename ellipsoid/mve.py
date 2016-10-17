"""Minimum Volume Ellipsoid classes."""

import numpy as np
import pandas as pd
from scipy import stats as st

# ------------------------------------------------------------------------------
# MINIMUM VOLUME ELLIPSOID ANALYSIS
# ------------------------------------------------------------------------------

class mve(object):

    """Base Minimum Volume Ellipsoid Analysis class.
    Implementation of TO BE COMPLETED.
    """

    def __init__(
        self,
        jump_singular_samples=True,
        number_of_samples=10000
    ):
        """Copy params to object properties, no validation."""
        self.jump_singular_samples = jump_singular_samples
        self.number_of_samples = number_of_samples

    def get_params(self, deep=True):
        """Return parameters as a dictionary."""
        params = {
            'jump_singular_samples': self.jump_singular_samples,
            'number_of_samples': self.number_of_samples
        }
        return params

    def set_params(self, **parameters):
        """Set parameters using kwargs."""
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

    def sampling(self, df, jump_singular_samples, number_of_samples):

        sing_count = 0
        self.deg_freedom = len(df.columns.values) - 1

        print 'Starting sampling..'
        for x in xrange(0, number_of_samples):
            s = df.sample(n=p+1)

            mean = np.array([s[col].mean() for col in s])
            cov_s = np.array(s.cov())
            mj = []
            pjs = []
            sample_indices = {}

            ind = [sum(i) for i in cov_s]
            ind = [i for i, j in enumerate(ind) if j == 0]
            for i in ind:
                cov_s[i, i] = 0.5

            if jump_singular_samples:

                if np.linalg.det(cov_s) != 0:
                    inv_cov = np.linalg.inv(cov_s)
                else:
                    sing_count += 1
            else:
                inv_cov = np.linalg.inv(cov_s)

            for obs in s.as_matrix():
                x = obs - mean
                mj.append(np.dot(np.dot(x, inv_cov), x.T))

            mj = np.median(np.array(mj))
            pjs.append(np.sqrt(np.linalg.det(np.dot(mj, cov_s))))

            sample_indices[str(pjs[-1])] = [s.index.values, mj, cov_s]
        print 'sampling completed. number of jumped samples due to singularity:'
        print sing_count

        min_sample = df.loc[sample_indices[str(min(pjs))][0], ]
        min_mj = sample_indices[str(min(pjs))][1]
        min_cov = np.array(sample_indices[str(min(pjs))][2])

        self.TX = np.array([min_sample[col].mean() for col in min_sample])

        c2 = np.square((1 + (15.0/df.shape[0]-(self.deg_freedom))))

        # median of chi2 distribution
        # chi2_med = (p*((1-(2.0/(9*p)))**3))
        chi2_med = st.chi2.median(self.deg_freedom, loc=0, scale=1)

        self.CX_inv = np.linalg.inv(np.dot((c2*(chi2_med**(-1)*min_mj), min_cov)))




    def weights(self, df):
        df_arr = np.array(df)

        W = []
        for obs in df_arr:
            x = obs - self.TX
            W.append(np.dot(np.dot(x, self.CX_inv), x.T))

        return W

    def outlier(self, W, quantile):
        outlier = []
        c_value = st.chi2.isf(q=quantile, df=self.deg_freedom)
        for i in W:
            if i > c_value:
                outlier.append(1)
            else:
                outlier.append(0)
        return outlier
