import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from AdalineAndLR import AdalineGD
from AdalineAndLR import LogisticRegressionGD

"""
Intro to Machine Learning Assignment 1
Encompasses the solution to Task 2.
Students: Jackie Javier, Pranitha Achanta
Built from code provided by the textbook (pg42).
"""
# Importing example data
iris = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data',
                 header = None)
wine = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data',
                 header = None)

# Use first 100 samples, setosa + versicolor (need binary)
y_iris = iris.iloc[0:100, 4].values
# Use first 129 samples, again to keep binary
y_wine = wine.iloc[0:129, 0].values

# Convert labels to {0, 1}
y_iris = np.where(y_iris == 'Iris-setosa', 0, 1)
y_wine = np.where(y_wine == 1, 0, 1)

# sepal length, sepal width, petal length, petal width
X_iris = iris.iloc[0:100, 0:4].values
# Ommitting the labels (0)
X_wine = wine.iloc[0:129, 1:14].values

# Standardization
X_iris_std = (X_iris - X_iris.mean(axis=0)) / X_iris.std(axis=0)
X_wine_std = (X_wine - X_wine.mean(axis=0)) / X_wine.std(axis=0)

# Establishing number of epochs and learning rate value
test_n = 100
test_eta = 0.01

# Iris
ada_iris = AdalineGD(n_iter=test_n, eta=test_eta).fit(X_iris_std, y_iris)
lr_iris = LogisticRegressionGD(n_iter=test_n, eta=test_eta).fit(X_iris_std, y_iris)

# Wine
ada_wine = AdalineGD(n_iter=test_n, eta=test_eta).fit(X_wine_std, y_wine)
lr_wine = LogisticRegressionGD(n_iter=test_n, eta=test_eta).fit(X_wine_std, y_wine)

# Figures
plt.figure(figsize=(6, 4))
plt.plot(ada_iris.losses_, label='AdalineGD Iris')
plt.plot(ada_wine.losses_, label='AdalineGD Wine')
plt.xlabel('Epochs')
plt.ylabel('MSE Loss')
plt.title('AdalineGD Iris and Wine Loss Convergence Comparison')
plt.legend()
plt.tight_layout()
plt.show()
plt.savefig("task2_ada.png")

plt.figure(figsize=(6, 4))
plt.plot(lr_iris.losses_, label='LogisticRegressionGD Iris')
plt.plot(lr_wine.losses_, label='LogisticRegressionGD Wine')
plt.xlabel('Epochs')
plt.ylabel('log Loss')
plt.title('LogisticRegressionGD Iris and Wine Loss Convergence Comparison')
plt.legend()
plt.tight_layout()
plt.show()
plt.savefig("task2_lr.png")

plt.figure(figsize=(6, 4))
plt.plot(ada_iris.losses_, label='AdalineGD Iris')
plt.plot(ada_wine.losses_, label='AdalineGD Wine')
plt.plot(lr_iris.losses_, label='LogisticRegressionGD Iris')
plt.plot(lr_wine.losses_, label='LogisticRegressionGD Wine')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('AdalineGD and LogisticRegressionGD Loss Convergence Comparison')
plt.legend()
plt.tight_layout()
plt.show()
plt.savefig("task2_compare.png")
