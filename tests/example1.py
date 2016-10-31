from sklearn.datasets import load_boston

import matplotlib.pyplot as plt


X1 = load_boston()['data'][:, [8, 10]]  # two clusters
X2 = load_boston()['data'][:, [5, 12]]  # "banana"-shaped


print plt.scatter(X1.transpose()[0], X1.transpose()[1])
plt.show()



