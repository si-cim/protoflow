# ProtoFlow

ProtoFlow is a TensorFlow-based Python toolbox for bleeding-edge research in prototype-based machine learning algorithms.

![tests](https://github.com/si-cim/protoflow/workflows/tests/badge.svg)
[![docs](https://readthedocs.org/projects/protoflow/badge/?version=latest)](https://protoflow.readthedocs.io/en/latest/?badge=latest)
[![PyPI](https://img.shields.io/pypi/v/protoflow)](https://pypi.org/project/protoflow/)
![PyPI - Downloads](https://img.shields.io/pypi/dm/protoflow?color=blue)
[![GitHub license](https://img.shields.io/github/license/si-cim/protoflow)](https://github.com/si-cim/protoflow/blob/master/LICENSE)

*PyTorch users, please see:* [ProtoTorch](https://github.com/si-cim/prototorch)

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

ProtoFlow can be easily installed using `pip`.
```
pip install -U protoflow
```
To also install the extras, use
```bash
pip install -U protoflow[examples,other,tests]
```
To install the bleeding-edge features and improvements:
```bash
git clone https://github.com/si-cim/prototorch.git
git checkout dev
cd prototorch
pip install -e .
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

## Bibtex

If you would like to cite the package, please use this:
```bibtex
@misc{Ravichandran2020a,
  author = {Ravichandran, J},
  title = {ProtoFlow},
  year = {2020},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/si-cim/protoflow}}
}
