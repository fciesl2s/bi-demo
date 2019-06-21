import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.datasets import make_moons

plt.figure()

n_samples = 300
X, y = make_blobs(n_samples=n_samples, centers=2)
y_pred = KMeans(n_clusters=2).fit_predict(X)

plt.scatter(X[:, 0], X[:, 1], c=y_pred)
plt.title("Erstes Beispiel")
plt.show()

# Nummer 2
n_samples = 300
X, y = make_blobs(n_samples=n_samples, centers=2)
y_pred = KMeans(n_clusters=3).fit_predict(X)

plt.scatter(X[:, 0], X[:, 1], c=y_pred)
plt.title("Zweites Beispiel")
plt.show()

# Nummer 3
n_samples = 300
X, y = make_moons(200, noise=.05, random_state=0)
y_pred = KMeans(n_clusters=2).fit_predict(X)

plt.scatter(X[:, 0], X[:, 1], c=y_pred)
plt.title("Drittes Beispiel")
plt.show()