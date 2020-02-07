# ProtoFlow

ProtoFlow is a TensorFlow-based Python toolbox for bleeding-edge research in prototype-based machine learning algorithms.

![Tests](https://github.com/si-cim/protoflow/workflows/Tests/badge.svg?branch=master)
[![Documentation Status](https://readthedocs.org/projects/protoflow/badge/?version=latest)](https://protoflow.readthedocs.io/en/latest/?badge=latest)


## Description

This is a Python toolbox brewed at the Mittweida University of Applied Sciences
in Germany for bleeding-edge research in Learning Vector Quantization (LVQ)
methods. Although, there are other (perhaps more extensive) LVQ toolboxes
available out there, the focus of ProtoFlow is ease-of-use, extensibility and
speed.

Many popular prototype-based Machine Learning (ML) algorithms like K-Nearest
Neighbors (KNN), Generalized Learning Vector Quantization (GLVQ) and Generalized
Matrix Learning Vector Quantization (GMLVQ) including the recent Learning Vector
Quantization Multi-Layer Network (LVQMLN) are implemented as Tensorflow models
using the Keras API.

## Installation

ProtoFlow can be installed using `pip`.
```
pip install protoflow
```

## Usage

ProtoFlow is modular. It is very easy to use the modular pieces provided by
ProtoFlow, like the layers, losses, callbacks and metrics to build your own
prototype-based(instance-based) models. These pieces blend-in seamlessly with
Keras allowing you to mix and match the modules from ProtoFlow with other Keras
modules.

ProtoFlow comes prepackaged with many popular LVQ algorithms in a convenient API,
with more algorithms and techniques coming soon. If you would simply like to be
able to use those algorithms to train large ML models on a GPU, ProtoFlow lets
you do this without requiring a black-belt in high-performance Tensor computation.
