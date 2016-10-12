"""Minimum Volume Ellipsoid classes."""

import numpy as np
import pandas as pd
from scipy import stats as st

# ------------------------------------------------------------------------------
# MINIMUM VOLUME ELLIPSOID ANALYSIS
# ------------------------------------------------------------------------------


class BaseMVE:

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


class MVE(BaseMVE):

    """
    """

    def __init__(self, **kwargs):
        """Copy params to object properties, no validation."""
        super(MVE, self).__init__(**kwargs)

    def get_params(self, deep=True):
        """Return parameters as a dictionary."""
        params = super(MVE, self).get_params(deep=deep)
        return params

    def sampling(df, jump_singular_samples, number_of_samples):

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

                try:
                    inv_cov = np.linalg.inv(cov_s)
                except:
                    sing_count += 1
                    continue
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

        TX = np.array([min_sample[col].mean() for col in min_sample])

        c2 = np.square((1 + (15.0/df.shape[0]-(self.deg_freedom))))

        # median of chi2 distribution
        # chi2_med = (p*((1-(2.0/(9*p)))**3))
        chi2_med = st.chi2.median(self.deg_freedom, loc=0, scale=1)

        CX_inv = np.linalg.inv(np.dot((c2*(chi2_med**(-1)*min_mj), min_cov))

        return (TX, CX_inv)

    def weights(df, TX, CX_inv):
        df_arr = np.array(df)

        W = []
        for obs in df_arr:
            x = obs - TX
            W.append(np.dot(np.dot(x, CX_inv), x.T))

        return W

    def outlier(W, q, p):
        outlier = []
        c_value = st.chi2.isf(q=q, df=p)
        for i in W:
            if i > c_value:
                outlier.append(1)
            else:
                outlier.append(0)
        return outlier
