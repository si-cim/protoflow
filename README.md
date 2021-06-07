# ProtoFlow: Prototype Learning in TensorFlow

![ProtoFlow Logo](https://protoflow.readthedocs.io/en/latest/_static/horizontal-lockup.png)

[![Build Status](https://travis-ci.org/si-cim/protoflow.svg?branch=master)](https://travis-ci.org/si-cim/protoflow)
![tests](https://github.com/si-cim/protoflow/workflows/tests/badge.svg)
[![GitHub tag (latest by date)](https://img.shields.io/github/v/tag/si-cim/protoflow?color=yellow&label=version)](https://github.com/si-cim/protoflow/releases)
[![docs](https://readthedocs.org/projects/protoflow/badge/?version=latest)](https://protoflow.readthedocs.io/en/latest/?badge=latest)
[![PyPI](https://img.shields.io/pypi/v/protoflow)](https://pypi.org/project/protoflow/)
[![codecov](https://codecov.io/gh/si-cim/protoflow/branch/master/graph/badge.svg)](https://codecov.io/gh/si-cim/protoflow)
![PyPI - Downloads](https://img.shields.io/pypi/dm/protoflow?color=blue)
[![GitHub license](https://img.shields.io/github/license/si-cim/protoflow)](https://github.com/si-cim/protoflow/blob/master/LICENSE)

**This project is no longer actively maintained**. Please consider using
[ProtoTorch](https://github.com/si-cim/prototorch) instead.

## Description

This is a Python toolbox brewed at the Mittweida University of Applied Sciences
in Germany for bleeding-edge research in Prototype-based Machine Learning
methods and other interpretable models. The focus of ProtoFlow is ease-of-use,
extensibility and speed.

## Installation

ProtoFlow can be easily installed using `pip`. To install the latest version, run
```
pip install -U protoflow
```
To also install the extras, run
```bash
pip install -U protoflow[all]
```

*Note: If you're using [ZSH](https://www.zsh.org/), the square brackets `[ ]`
have to be escaped like so: `\[\]`, making the install command `pip install -U
prototorch\[all\]`.*

To install the bleeding-edge features and improvements before they are release on PyPI, run
```bash
git clone https://github.com/si-cim/protoflow.git
cd protoflow
git checkout dev
pip install -e .[all]
```

For gpu support, additionally run
```bash
pip install -U protoflow[gpu]
```
or simply install `tensorflow-gpu` manually.

## Documentation

The documentation is available at <https://www.protoflow.ml/en/latest/>. Should
that link not work try <https://protoflow.readthedocs.io/en/latest/>.

## Usage

### For researchers
ProtoFlow is modular. It is very easy to use the modular pieces provided by
ProtoFlow, like the layers, losses, callbacks and metrics to build your own
prototype-based(instance-based) models. These pieces blend-in seamlessly with
Keras allowing you to mix and match the modules from ProtoFlow with other Keras
modules.

### For engineers
ProtoFlow comes prepackaged with many popular Learning Vector Quantization
(LVQ)-like algorithms in a convenient API. If you would simply like to be able
to use those algorithms to train large ML models on a GPU, ProtoFlow lets you do
this without requiring a black-belt in high-performance Tensor computation.

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
