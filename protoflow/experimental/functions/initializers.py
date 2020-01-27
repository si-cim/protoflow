"""ProtoFlow weight initializers."""

import numpy as np


def get_random_prototype_initializer(x_train,
                                     y_train,
                                     prototype_distribution=None):
    labels = np.unique(y_train)
    if not prototype_distribution:
        prototype_distribution = [1] * len(labels)

    def random_prototype_initializer():
        protos = np.empty(shape=(0, x_train.shape[1]), dtype=x_train.dtype)
        proto_labels = np.empty(shape=(0, ), dtype=np.int32)
        for label, num in zip(labels, prototype_distribution):
            x_label = x_train[y_train == label]
            # print(f'Found {x_label.shape[0]} samples of class {label}.')
            # print(f'Selecting {num} samples of class {label}.')
            indices = list(range(x_label.shape[0]))
            chosen_indices = np.random.choice(indices, size=num, replace=False)
            protos = np.append(protos, x_label[chosen_indices], axis=0)
            proto_labels = np.append(proto_labels, [label] * num)
        return protos

    return random_prototype_initializer


def get_classmeans_prototype_initializer(x_train,
                                         y_train,
                                         prototype_distribution=None):
    labels = np.unique(y_train)
    if not prototype_distribution:
        prototype_distribution = [1] * len(labels)

    def classmeans_prototype_initializer():
        protos = np.empty(shape=(0, x_train.shape[1]), dtype=x_train.dtype)
        proto_labels = np.empty(shape=(0, ), dtype=np.int32)
        for label, num in zip(labels, prototype_distribution):
            x_label = x_train[y_train == label]
            # print(f'Found {x_label.shape[0]} samples of class {label}.')
            # print(f'Selecting {num} samples of class {label}.')
            x_label_mean = np.mean(x_label, axis=0)
            x_label_mean = x_label_mean.reshape(1, x_train.shape[1])
            for _ in range(num):
                protos = np.append(protos, x_label_mean, axis=0)
            proto_labels = np.append(proto_labels, [label] * num)
        return protos

    return classmeans_prototype_initializer
