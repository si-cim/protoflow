import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

import protoflow as pf
from protoflow import experimental as pfe


def basic_sanity_check(x_train, y_train):
    # Basic sanity checks
    if len(x_train.shape) != 2:
        raise ValueError('x_train has to be of shape (n, m).')
    if len(y_train.shape) != 1:
        raise ValueError('y_train has to be of shape (n, ).'
                         f'You provide shape: {y_train.shape}.')
    if True in np.isnan(x_train).any(axis=1):
        raise ValueError('NaN in x_train.')
    if True in np.isnan(y_train):
        raise ValueError('NaN in y_train.')
    if 0 not in y_train:
        raise ValueError('Labels required to be zero-indexed.'
                         f'You provide: {np.unique(y_train)}.')
    if np.max(y_train) >= np.unique(y_train).size:
        raise ValueError('Labels required to be linearly spaced.'
                         f'You provide: {np.unique(y_train)}.')


def get_basic_glvq(x_train, y_train, lr=0.01, normalize=True):
    basic_sanity_check(x_train, y_train)
    if normalize:
        scaler = StandardScaler()
        scaler.fit(x_train)
        x_train = scaler.transform(x_train)
    num_classes = np.unique(y_train).size
    protos, labels = pf.utils.classwise_mean(x_train, y_train,
                                             [1] * num_classes)
    dist = pf.layers.SquaredEuclideanDistance(protos.shape[0],
                                              protos.shape[1],
                                              labels,
                                              custom_prototypes=protos)
    wtac = pf.layers.WTAC(labels)
    model = tf.keras.models.Sequential([dist])
    model.compile(
        loss=pf.losses.GLVQLoss(labels),
        optimizer=tf.keras.optimizers.Adam(lr),
        metrics=[pf.metrics.wtac_accuracy(labels)],
    )
    clf = tf.keras.models.Sequential([model, wtac])
    if scaler:
        return model, clf, scaler
    else:
        return model, clf


def get_2d_lvqmln(x_train,
                  y_train,
                  activation='sigmoid',
                  lr=0.01,
                  normalize=False,
                  video=False,
                  resolution=50):
    basic_sanity_check(x_train, y_train)
    if normalize:
        scaler = StandardScaler()
        scaler.fit(x_train)
        x_train = scaler.transform(x_train)
    num_classes = np.unique(y_train).size
    protos, labels = pf.utils.classwise_mean(x_train, y_train,
                                             [1] * num_classes)
    proj = tf.keras.layers.Dense(2, use_bias=False, activation=activation)
    protos = proj(protos)
    dist = pf.layers.SquaredEuclideanDistance(protos.shape[0],
                                              protos.shape[1],
                                              labels,
                                              custom_prototypes=protos)
    wtac = pf.layers.WTAC(labels)
    model = tf.keras.models.Sequential([proj, dist])
    model.compile(
        loss=pf.losses.GLVQLoss(labels),
        optimizer=tf.keras.optimizers.Adam(lr),
        metrics=[pf.metrics.wtac_accuracy(labels)],
    )
    clf = tf.keras.models.Sequential([model, wtac])
    dvis = pfe.callbacks.VisPointProtos(data=np.c_[x_train, y_train],
                                        interval=1,
                                        voronoi=True,
                                        show=not (video),
                                        resolution=resolution,
                                        snap=video,
                                        make_mp4=video,
                                        cmap='plasma')
    model._projection = proj
    model.competition = wtac
    if normalize:
        return model, clf, dvis, scaler
    else:
        return model, clf, dvis
