"""Fit KNN models on the Iris dataset."""

import numpy as np
from matplotlib import pyplot as plt
from sklearn import datasets
from sklearn.metrics import accuracy_score

from protoflow.applications import KNN
from protoflow.utils import color_scheme

iris = datasets.load_iris()
sel_features = [0, 3]
x = np.array([i[sel_features] for i in iris.data])
y = np.array(iris.target)

flower_labels = ["iris setosa", "iris virginica", "iris versicolor"]

# Normalize the x data to the range 0 to 1
x = (x - x.min(axis=0)) / (x.max(axis=0) - x.min(axis=0))

# Create indices for the train-test split
np.random.seed(42)
split = 0.5
train_indices = np.random.choice(len(x), round(len(x) * split), replace=False)
test_indices = np.array(list(set(range(len(x))) - set(train_indices)))

# The train-test split
x_train = x[train_indices].astype('float')
x_test = x[test_indices].astype('float')
y_train = y[train_indices].astype('int')
y_test = y[test_indices].astype('int')

n = 8
cmap = 'plasma'
plt.figure('Effect of K')
for i in range(1, n + 1):
    model = KNN(k=i)
    model.fit(x_train, y_train)
    y_pred = model.predict(x_train)
    acc = accuracy_score(y_train, y_pred)
    print(f'Train accuracy of KNN with K={i} is {(acc * 100):6.04f}%')
    y_pred = model.predict(x_test)
    acc = accuracy_score(y_test, y_pred)
    print(f'Test accuracy of KNN with K={i} is  {(acc * 100):6.04f}%')
    plt.subplot(2, n / 2, i)
    plt.title(f'K={i}')
    ax = plt.gca()
    ax.axis('off')
    border = 0.05
    resolution = 70
    x_min, x_max = x_test[:, 0].min() - border, x_test[:, 0].max() + border
    y_min, y_max = x_test[:, 1].min() - border, x_test[:, 1].max() + border
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 1 / resolution),
                         np.arange(y_min, y_max, 1 / resolution))
    z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    z = z.reshape(xx.shape)
    color_cycle = color_scheme(3, cmap=cmap)
    y_test_colors = list(map(lambda y: color_cycle[y + 1], y_test))
    plt.contourf(xx, yy, z, cmap=plt.get_cmap(cmap), alpha=0.35)
    plt.scatter(x_test[:, 0], x_test[:, 1], c=y_test_colors, edgecolors='k')
plt.show()
