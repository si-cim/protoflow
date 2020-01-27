"""ProtoFlow data utilities."""

import protoflow as pf


def classwise_random(x_train, y_train, prototype_distribution, verbose=False):
    """Randomly sample data by class."""
    initializer = pf.initializers.get_classwise_random_initializer(
        x_train, y_train, prototype_distribution, verbose)
    shape = (sum(prototype_distribution), x_train.shape[1])
    prototypes = initializer(shape)
    prototype_labels = initializer.prototype_labels
    return prototypes, prototype_labels


def classwise_mean(x_train, y_train, prototype_distribution, verbose=False):
    """Sample data means by class."""
    initializer = pf.initializers.get_classwise_mean_initializer(
        x_train, y_train, prototype_distribution, verbose)
    shape = (sum(prototype_distribution), x_train.shape[1])
    prototypes = initializer(shape)
    prototype_labels = initializer.prototype_labels
    return prototypes, prototype_labels
