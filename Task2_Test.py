import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from AdalineAndLR import AdalineGD
from AdalineAndLR import LogisticRegressionGD

"""
Intro to Machine Learning Assignment 1
Encompasses the solution to Task 2.
Students: Jackie Javier, e.t.c.
Built from code provided by the textbook (pg42).
"""
# Importing example data 
# (no longer found at given links from the book)
iris = pd.read_csv('data/iris.data',
                 header = None)
wine = pd.read_csv('data/wine.data',
                 header = None)

# Use first 100 samples, setosa + versicolor (need binary)
y = iris.iloc[0:100, 4].values
# Use first 100 samples, again to keep binary (?)
y2 = wine.iloc[0:100, 0].values

# Convert labels to {0, 1}
y = np.where(y == 'Iris-setosa', 0, 1)
y2 = np.where(y == 1, 0, 1)

# sepal length, sepal width, petal length, petal width
X = iris.iloc[0:100, 0:4].values
# Ommitting the labels (0)
X2 = wine.iloc[0:100, 1:14].values

fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(10, 4))

# Iris
ada1 = AdalineGD(n_iter=15, eta=0.1).fit(X, y)
ax[0][0].plot(
    range(1, len(ada1.losses_) + 1),
    np.log10(ada1.losses_),
    marker='o'
)
ax[0][0].set_xlabel('Epochs')
ax[0][0].set_ylabel('log(Mean squared error)')
ax[0][0].set_title('AdalineGD Iris (eta=0.1)')

lr1 = LogisticRegressionGD(n_iter=15, eta=0.1).fit(X, y)
ax[0][1].plot(
    range(1, len(lr1.losses_) + 1),
    np.log10(lr1.losses_),
    marker='o'
)
ax[0][1].set_xlabel('Epochs')
ax[0][1].set_ylabel('log(Change in magnitude)')
ax[0][1].set_title('LogisticRegressionGD Iris (eta=0.1)')

# Wine
ada2 = AdalineGD(n_iter=15, eta=0.1).fit(X2, y2)
ax[1][0].plot(
    range(1, len(ada2.losses_) + 1),
    np.log10(ada2.losses_),
    marker='o'
)
ax[1][0].set_xlabel('Epochs')
ax[1][0].set_ylabel('log(Mean squared error)')
ax[1][0].set_title('AdalineGD Wine (eta=0.1)')

lr2 = LogisticRegressionGD(n_iter=15, eta=0.1).fit(X2, y2)
ax[1][1].plot(
    range(1, len(lr2.losses_) + 1),
    np.log10(lr2.losses_),
    marker='o'
)
ax[1][1].set_xlabel('Epochs')
ax[1][1].set_ylabel('log(Change in magnitude)')
ax[1][1].set_title('LogisticRegressionGD Wine (eta=0.1)')

plt.show()