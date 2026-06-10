import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from AdalineAndLR import AdalineGD

"""
Intro to Machine Learning Assignment 1
Encompasses the solution to Task 3.
Students: Jackie Javier, Pranitha Achanta, Robert McDaniels
"""
# Preparing the data
iris = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data',
                 header = None)

y = iris.iloc[:,4].values
X = iris.iloc[:,0:4].values

# Desired label = 1, rest = 0
y_setosa = np.where(y == 'Iris-setosa', 1, 0)
y_versicolor = np.where(y == 'Iris-versicolor', 1, 0)
y_virginica = np.where(y == 'Iris-virginica', 1, 0)

# Standardize
X_std = (X - X.mean(axis=0)) / X.std(axis=0)

# Establishing number of epochs and learning rate value
test_n = 500
test_eta = 0.001

# Setosa
ada_setosa = AdalineGD(n_iter=test_n, eta=test_eta).fit(X_std, y_setosa)

# Versicolor
ada_versicolor = AdalineGD(n_iter=test_n, eta=test_eta).fit(X_std, y_versicolor)

# Virginica
ada_virginica = AdalineGD(n_iter=test_n, eta=test_eta).fit(X_std, y_virginica)

# Comparing real y to predicted y
X_bias = np.hstack([X_std, np.ones((X_std.shape[0], 1))])
setosa = ada_setosa.net_input(X_bias)
versicolor = ada_versicolor.net_input(X_bias)
virginica = ada_virginica.net_input(X_bias)

results = np.column_stack((setosa, versicolor, virginica))
row_max = np.max(results, axis=1, keepdims=True)
one_hot = (results == row_max).astype(int)
compare = np.column_stack((y, one_hot))
print(compare)

# Figure
plt.figure(figsize=(6, 4))
plt.plot(ada_setosa.losses_, label='Setosa')
plt.plot(ada_versicolor.losses_, label='Versicolor')
plt.plot(ada_virginica.losses_, label='Virginica')
plt.xlabel('Epochs')
plt.ylabel('MSE Loss')
plt.title('AdalineGD Iris Loss Convergence Comparison')
plt.legend()
plt.tight_layout()
plt.savefig("report/task3_compare.png")
