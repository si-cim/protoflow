"""Fit an LVQMLN model to the Iris dataset."""

import numpy as np
import tensorflow as tf
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler

from protoflow.applications import LVQMLN
from protoflow.experimental.callbacks import VisPointProtos

# Set hyperparameters
BATCH_SIZE = 8
EPOCHS = 30
LR = 0.01

# Prepare and preprocess the data
scaler = StandardScaler()
x_train, y_train = load_iris(True)
x_train = x_train[:, [0, 2]].astype('float')
y_train = y_train.astype('int')
scaler.fit(x_train)
x_train = scaler.transform(x_train)

clf = LVQMLN(
    projection_dim=2,
    prototype_distribution=[1, 2, 3],
    activation='sigmoid',
)

# Callbacks
es = tf.keras.callbacks.EarlyStopping(
    monitor='loss',
    mode='min',
    patience=5,
    restore_best_weights=True,
)
dvis = VisPointProtos(
    data=np.c_[x_train, y_train],
    voronoi=True,
    project_mesh=False,
    border=0.15,
    resolution=30,
    show=True,
    snap=False,
    make_mp4=False,
)

# Train with the visualization callback to observe the learning
clf.fit(x_train,
        y_train,
        callbacks=[
            es,
            dvis,
        ],
        verbose=True,
        epochs=EPOCHS,
        lr=LR,
        batch_size=BATCH_SIZE)

# Accuracy
print('Training accuracy after training:')
clf.score(x_train, y_train, verbose=True)
