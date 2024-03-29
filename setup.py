"""Install ProtoFlow."""

from setuptools import setup
from setuptools import find_packages

VERSION = "0.3.4"
PROJECT_URL = "https://github.com/si-cim/protoflow"
DOWNLOAD_URL = "{}/releases/tag/v{}".format(PROJECT_URL, VERSION)

with open("README.md", "r") as fh:
    LONG_DESCRIPTION = fh.read()

INSTALL_REQUIRES = [
    "tensorflow>=2.3.1",
    "scikit-learn>=0.23.2",
    "matplotlib>=3.3.2",
    "requests>=2.24.0",
    "tqdm>=4.51.0",
]
DOCS = [
    "recommonmark",
    "sphinx",
    "sphinx_rtd_theme",
    "sphinxcontrib-katex",
]
GPU = ["tensorflow-gpu"]
OTHERS = [
    "xlrd",
    "pandas",
    "pydot",
    "seaborn",
    "imageio",
]
TESTS = ["pytest"]
ALL = DOCS + OTHERS + TESTS

setup(name="protoflow",
      version=VERSION,
      description="Highly extensible, GPU-supported "
      "Learning Vector Quantization (LVQ) toolbox "
      "built using Tensorflow 2.x and its Keras API.",
      long_description=LONG_DESCRIPTION,
      long_description_content_type="text/markdown",
      author="Jensun Ravichandran",
      author_email="jjensun@gmail.com",
      url=PROJECT_URL,
      download_url=DOWNLOAD_URL,
      license="MIT",
      install_requires=INSTALL_REQUIRES,
      extras_require={
          "docs": DOCS,
          "gpu": GPU,
          "others": OTHERS,
          "tests": TESTS,
          "all": ALL,
      },
      classifiers=[
          "Development Status :: 3 - Alpha",
          "Environment :: Console",
          "Intended Audience :: Developers",
          "Intended Audience :: Education",
          "Intended Audience :: Science/Research",
          "License :: OSI Approved :: MIT License",
          "Natural Language :: English",
          "Programming Language :: Python :: 3.6",
          "Programming Language :: Python :: 3.7",
          "Programming Language :: Python :: 3.8",
          "Operating System :: OS Independent",
          "Topic :: Scientific/Engineering :: Artificial Intelligence",
          "Topic :: Software Development :: Libraries",
          "Topic :: Software Development :: Libraries :: Python Modules",
      ],
      packages=find_packages())
