from sklearn.datasets import load_boston

import matplotlib.pyplot as plt
import numpy

from ellipsoid.mve import mve

numpy.random.seed(12345)

mymve = mve()

mymve.set_params(n_samples=5000)

# load dataset
#X1 = load_boston()['data'][:, [8, 10]]  # two clusters
X1 = load_boston()['data'][:, [5, 12]]  # "banana"-shaped

# fit MVE
mymve.fit(X1)
# get ranking
distances = mymve.get_distances()


# ANALYSE RESULTS GRAPHICALLY
percentile = 0.74

binary_distances = distances <= numpy.percentile(distances, percentile * 100, axis=0)


plt.scatter(X1.transpose()[0][binary_distances], X1.transpose()[1][binary_distances], color='b')
plt.scatter(X1.transpose()[0][~binary_distances], X1.transpose()[1][~binary_distances], color='r')

'''
for i, txt in enumerate(distances):
    if txt >= numpy.percentile(distances, percentile * 100, axis=0):
        plt.annotate(txt, (X1.transpose()[0][i],X1.transpose()[1][i]))
'''

#plt.colorbar()
#plt.scatter(X1.transpose()[0, binary_distances], X1.transpose()[1, binary_distances])
plt.show()

