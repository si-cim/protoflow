"""Install ProtoFlow."""

from setuptools import setup
from setuptools import find_packages

PROJECT_URL = 'https://github.com/si-cim/protoflow'
DOWNLOAD_URL = 'https://github.com/si-cim/protoflow.git'

with open('README.md', 'r') as fh:
    long_description = fh.read()

setup(name='protoflow',
      version='0.2.1',
      description='Highly extensible, GPU-supported '
      'Learning Vector Quantization (LVQ) toolbox '
      'built using Tensorflow 2.x and its Keras API.',
      long_description=long_description,
      long_description_content_type='text/markdown',
      author='Jensun Ravichandran',
      author_email='jjensun@gmail.com',
      url=PROJECT_URL,
      download_url=DOWNLOAD_URL,
      license='MIT',
      install_requires=[
          'tensorflow>=2.2.0',
          'numpy>=1.9.1',
          'sklearn',
      ],
      extras_require={
          'examples': ['matplotlib'],
          'other': [
              'xlrd',
              'pandas',
              'seaborn',
              'imageio',
          ],
          'tests': ['pytest'],
          'docs': [
              'sphinx',
              'sphinx_rtd_theme',
              'sphinxcontrib-katex',
          ],
      },
      classifiers=[
          'Development Status :: 2 - Pre-Alpha', 'Environment :: Console',
          'Intended Audience :: Developers', 'Intended Audience :: Education',
          'Intended Audience :: Science/Research',
          'License :: OSI Approved :: MIT License',
          'Programming Language :: Python :: 3.6',
          'Programming Language :: Python :: 3.7',
          'Programming Language :: Python :: 3.8',
          'Operating System :: OS Independent',
          'Topic :: Scientific/Engineering :: Artificial Intelligence',
          'Topic :: Software Development :: Libraries',
          'Topic :: Software Development :: Libraries :: Python Modules'
      ],
      packages=find_packages())
