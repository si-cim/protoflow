"""Fit a GLVQ model to the Iris dataset.

If you're having trouble running this example script on Windows, remove the
visualization callback.
"""

import tensorflow as tf
from sklearn.datasets import load_iris

import protoflow as pf
from protoflow.callbacks import VisPointProtos

# Prepare and preprocess the data
x_train, y_train = load_iris(return_X_y=True)
x_train = x_train[:, [0, 2]]

# Build the model
model = pf.applications.GLVQ(nclasses=len(set(y_train)),
                             input_dim=x_train.shape[-1],
                             prototypes_per_class=1)

# Print summary
model.summary()

# Compile the model
model.compile(optimizer=tf.keras.optimizers.Adam(0.1))

# Callbacks
dvis = VisPointProtos(prototype_layer=model.prototype_layer,
                      data=(x_train, y_train),
                      snap=False,
                      voronoi=True,
                      resolution=50,
                      pause_time=0.1)

# Train with the visualization callback to observe the learning
model.fit(x_train,
          y_train,
          verbose=True,
          callbacks=[dvis],
          epochs=200,
          batch_size=32)
