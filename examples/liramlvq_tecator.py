"""Build a limited rank GMLVQ model for the Tecator dataset."""

import tensorflow as tf

import protoflow as pf
from protoflow.datasets.tecator import load_data

(x_train, y_train) = load_data()

x_train = x_train.astype('float')
y_train = y_train.astype('int')

dist = pf.layers.OmegaDistance(2, [0, 1],
                               mapping_dim=2,
                               matrix_scope='global',
                               input_dim=x_train.shape[1])
clf = tf.keras.Sequential([dist])

omega = dist.omega.numpy()
protos = dist.prototypes.numpy()
protos, proto_labels = pf.utils.data.classwise_mean(x_train, y_train, [1, 1])
dist.set_weights([omega, protos])

clf.compile(loss=pf.losses.GLVQLoss([0, 1]),
            optimizer=tf.keras.optimizers.Adam(0.001),
            metrics=[pf.metrics.wtac_accuracy([0, 1])])
clf.fit(x_train, y_train, verbose=True, epochs=300, batch_size=128)
