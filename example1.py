from sklearn.datasets import load_boston

import matplotlib.pyplot as plt

from ellipsoid.mve import mve
import pandas


X1 = load_boston()['data'][:, [8, 10]]  # two clusters
X2 = load_boston()['data'][:, [5, 12]]  # "banana"-shaped



mymve = mve()

mymve.sampling(pandas.DataFrame(X1), True, 50)

#plt.scatter(X1.transpose()[0], X1.transpose()[1])
#plt.show()



