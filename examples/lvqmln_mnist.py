"""Fit an LVQMLN model to the MNIST dataset."""

import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.datasets import mnist

from protoflow.applications import LVQMLN

# Set hyperparameters.
BATCH_SIZE = 64
EPOCHS = 100
LR = 1e-4

# Prepare and preprocess the data.
scaler = StandardScaler()
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(-1, np.prod(x_train.shape[1:]))
x_test = x_test.reshape(-1, np.prod(x_test.shape[1:]))

x_train = x_train.astype('float')
x_test = x_test.astype('float')
y_train = y_train.astype('int')
y_test = y_test.astype('int')

scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

print(f'x_train shape: {x_train.shape}')
print(f'y_train shape: {y_train.shape}')
print(f'Number of classes: {len(np.unique(y_train))}')

clf = LVQMLN(
    projection_dim=10,
    prototypes_per_class=1,
)

# Manually build the model.
clf.build(x_train, y_train, prototype_initializer='mean')

# Use the predict method of GLVQ to obtain class labels.
y_pred = clf.predict(x_train)

# Accuracy
print('Training accuracy before training:')
clf.score(x_train, y_train, verbose=True)

# Callbacks
es = tf.keras.callbacks.EarlyStopping(
    monitor='loss',
    mode='min',
    patience=5,
    restore_best_weights=True,
)

# Train
clf.fit(x_train,
        y_train,
        verbose=True,
        callbacks=[es],
        epochs=EPOCHS,
        lr=LR,
        batch_size=BATCH_SIZE)

# Accuracy
print('Training accuracy after training:')
clf.score(x_train, y_train, verbose=True)
print('Test accuracy after training:')
clf.score(x_test, y_test, verbose=True)
