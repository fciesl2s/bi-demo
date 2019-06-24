import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.datasets import make_moons

plt.figure()

# Beispiel 1
n_samples = 300
X, y = make_blobs(n_samples=n_samples, centers=3, random_state=200)

plt.scatter(X[:, 0], X[:, 1])
plt.title("Erstes Beispiel")
plt.show()

y_pred = KMeans(n_clusters=3).fit_predict(X)

plt.scatter(X[:, 0], X[:, 1], c=y_pred)
plt.title("Erstes Beispiel")
plt.show()

# Beispiel 2
n_samples = 300
X, y = make_blobs(n_samples=n_samples, centers=3, random_state=100)

plt.scatter(X[:, 0], X[:, 1])
plt.title("Zweites Beispiel")
plt.show()

y_pred = KMeans(n_clusters=4).fit_predict(X)

plt.scatter(X[:, 0], X[:, 1], c=y_pred)
plt.title("Zweites Beispiel")
plt.show()


# Beispiel 3
n_samples = 300
X, y = make_moons(n_samples=n_samples, noise=.05, random_state=0)

plt.scatter(X[:, 0], X[:, 1])
plt.title("Drittes Beispiel")
plt.show()

y_pred = KMeans(n_clusters=2).fit_predict(X)

plt.scatter(X[:, 0], X[:, 1], c=y_pred)
plt.title("Drittes Beispiel")
plt.show()

# Beispiel 4
n_samples = 300
X, y = make_blobs(n_samples=n_samples, centers=3, random_state=170)
transformation = [[0.60834549, -0.63667341], [-0.40887718, 0.85253229]]
X = np.dot(X, transformation)

plt.scatter(X[:, 0], X[:, 1])
plt.title("Viertes Beispiel")
plt.show()

y_pred = KMeans(n_clusters=3).fit_predict(X)

plt.scatter(X[:, 0], X[:, 1], c=y_pred)
plt.title("Viertes Beispiel")
plt.show()