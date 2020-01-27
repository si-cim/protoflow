"""Fit a GLVQ model to the Iris dataset."""

import numpy as np
import tensorflow as tf
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler

from protoflow.applications import GLVQ
from protoflow.experimental.callbacks import VisPointProtos

# Set hyperparameters
BATCH_SIZE = 8
EPOCHS = 10
LR = 0.01

# Prepare and preprocess the data
scaler = StandardScaler()
x_train, y_train = load_iris(True)
x_train = x_train[:, [0, 2]].astype('float')
y_train = y_train.astype('int')
scaler.fit(x_train)
x_train = scaler.transform(x_train)

clf = GLVQ(prototypes_per_class=1)

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
    project_protos=False,
    border=0.3,
    resolution=50,
    cmap='plasma',
    pause_time=0.1,
    show=False,
    save=False,
    snap=True,
    make_gif=False,
    make_mp4=True,
)

# Train with the visualization callback to observe the learning
clf.fit(x_train,
        y_train,
        verbose=True,
        callbacks=[es, dvis],
        epochs=EPOCHS,
        lr=LR,
        batch_size=BATCH_SIZE)

# Accuracy
print('Training accuracy after training:')
clf.score(x_train, y_train, verbose=True)
