"""Fit a GLVQ model to the Iris dataset using functional API of Keras.

If you're having trouble running this example script on Windows, remove the
visualization callback.
"""

import os

import tensorflow as tf
from sklearn.datasets import load_iris

import protoflow as pf
from protoflow.callbacks import VisPointProtos

# Prepare and preprocess the data
x_train, y_train = load_iris(return_X_y=True)
x_train = x_train[:, [0, 2]]

# Build the layers
inputs = tf.keras.layers.Input(shape=(x_train.shape[-1]),
                               name='x')  # no batch-size
prototype_layer = pf.layers.prototypes.AppendPrototypes1D(
    nclasses=len(set(y_train)),
    prototypes_per_class=5,
    name='w',
)
plabels = prototype_layer.prototype_labels.numpy()
euclidean = pf.layers.distances.Euclidean(plabels, name='d')
prototypes = prototype_layer(inputs)
distances = euclidean(prototypes)

# Build the model using the functional API
model = tf.keras.models.Model(inputs=inputs,
                              outputs=distances,
                              name="GLVQ_model")

# Print summary
model.summary()

# Plot model as pdf
tf.keras.utils.plot_model(model, "GLVQ_Iris_model.pdf", show_shapes=True)

# Warmstart prototypes
protoinit = pf.initializers.StratifiedRandom(
    x_train,
    y_train,
    epsilon=0.1,
    prototype_distribution=prototype_layer.prototype_distribution)
protos, plabels = protoinit(shape=prototype_layer.prototypes.shape)
prototype_layer.set_weights([protos])

# Compile the model
model.compile(loss=pf.losses.GLVQLossLabeless(),
              optimizer=tf.keras.optimizers.Adam(0.01),
              metrics=pf.modules.metrics.wtac_accuracy(),
              run_eagerly=False)

# Callbacks
dvis = VisPointProtos(prototype_layer=prototype_layer,
                      data=(x_train, y_train),
                      snap=False,
                      voronoi=False,
                      resolution=50,
                      pause_time=0.1)

# Train with the visualization callback to observe the learning
model.fit(x_train,
          y_train,
          verbose=True,
          callbacks=[dvis],
          epochs=20,
          batch_size=32)

# Save the model
if not os.path.exists("./saved"):
    os.makedirs("./saved")
model.save("./saved/glvq_iris.h5")

# Load the model
loaded_model = tf.keras.models.load_model(
    "./saved/glvq_iris.h5",
    custom_objects={
        "AppendPrototypes1D": pf.layers.prototypes.AppendPrototypes1D,
        "Euclidean": pf.layers.distances.Euclidean,
        "GLVQLossLabeless": pf.losses.GLVQLossLabeless,
    })

# Print summary
loaded_model.summary()
