"""Fit a GLVQ model to the Iris dataset using the Keras model API."""

import tensorflow as tf
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score

import protoflow as pf

x_train, y_train = load_iris(True)
x_train = x_train[:, [0, 2]].astype('float')
y_train = y_train.astype('int')

h1 = tf.keras.layers.Dense(units=3, input_dim=x_train.shape[1])
dist = pf.layers.EuclideanDistance(num_of_prototypes=3,
                                   prototype_labels=[0, 1, 2],
                                   input_dim=x_train.shape[1])
model = tf.keras.Sequential([h1, dist])

model.summary()

model.compile(loss=pf.losses.GLVQLoss(prototype_labels=[0, 1, 2], beta=10),
              optimizer=tf.keras.optimizers.Adam(0.01),
              metrics=[pf.metrics.wtac_accuracy(prototype_labels=[0, 1, 2])])
model.fit(
    x_train,
    y_train,
    epochs=200,
    callbacks=[tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)])

y_pred = model.predict(x_train).argmin(axis=1)
acc = accuracy_score(y_pred, y_train)
print('Train accuracy:', acc)
