"""Module to build and evaluate LVQ1 models with Keras."""

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from protoflow.applications.glvq import GLVQ


class LVQ1(GLVQ):
    """Learning Vector Quantization 1 (LVQ1).

    LVQ1, introduced by Kohonen is a heuristic learning
    scheme which is more stable than KNN.
    """

    # def build(self, *args, **kwargs):
    #     # Replace the competition layer.
    #     self.competition = layers.WTAC(
    #         prototype_labels=prototype_labels,
    #         dtype=self.dtype,
    #         name='wtac',
    #     )

    def fit(self,
            x_train,
            y_train,
            epochs=2,
            batch_size=32,
            learning_rate=0.01,
            shuffle=True,
            callbacks=[],
            run_eagerly=False,
            **kwargs):
        """Heuristically optimize the prototypes to fit the training data.

        Arguments:
            x_train : Training inputs.
            y_train : Training targets.

        Keyword Arguments:
            batch_size (int): Size of training batches. (default: 32)
            learning_rate (float): Learning rate (same as lr). (default: 0.01)
            shuffle (bool): Shuffle the training set. (default: True)
            callbacks (list): List of callbacks. (default: [])
            run_eagerly (bool): Use eager execution. (default: False)
        """
        learning_rate = kwargs.pop('lr', learning_rate)

        if not self.built:
            self.build(x_train, y_train)
        self._compile(
            learning_rate,
            run_eagerly=run_eagerly,
        )

        # Hook model to callbacks
        for cb in callbacks:
            cb.model = self.model

        num_batches = x_train.shape[0]
        # Train
        for cb in callbacks:
            cb.on_train_begin(logs=None)
        for i in range(epochs):
            logs = {}
            for cb in callbacks:
                cb.on_epoch_begin(epoch=i + 1, logs=logs)
            print(f'Epoch {i}/{epochs}')
            pbar = tf.keras.utils.Progbar(target=num_batches)

            if shuffle:
                data = np.c_[x_train, y_train]
                np.random.shuffle(data)
                x_train = data[:, :-1].astype(x_train.dtype)
                y_train = data[:, -1].astype(y_train.dtype)

            d = self.model(x_train)
            x_proj = self.projection(x_train).numpy()
            for j in range(num_batches):
                wj = tf.argmax(-d[j])  # winning index
                wl = self.prototype_labels[wj]  # winning label

                prototypes = self.distance.prototypes.numpy()
                if int(wl) == int(y_train[j]):
                    # Attraction
                    prototypes[wl] = prototypes[wl] + learning_rate * (
                        x_proj[j] - prototypes[wl])
                # else:
                #     # Repulsion
                #     prototypes[wl] = prototypes[wl] - learning_rate * (
                #         x_proj[j] - prototypes[wl])

                self.set_prototypes(prototypes)
                pbar.update(j + 1)

                acc = self.score(x_train, y_train)
                logs['acc'] = acc
                for cb in callbacks:
                    cb.on_batch_end(batch=j + 1, logs=logs)
                plt.show()
            for cb in callbacks:
                cb.on_epoch_end(epoch=i + 1, logs=logs)
        for cb in callbacks:
            cb.on_train_end(logs=None)

    @property
    def loss_history(self):
        """Override parent class property.

        Returns:
            None
        """
        return None
