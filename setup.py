from distutils.core import setup
from setuptools import find_packages

setup(
    name='SmarterThanBackProp',
    version='1.0',
    description='A package containg controlled based methods for training neural networks',
    author='Conrad Li',
    author_email='conradliste@utexas.edu',
    requires=[
        'numpy',
        'matplotlib',
        'torch',
        'jax'
        'flax'],
    packages=find_packages(),
    url='https://github.com/conradliste/SmarterThanBackProp'
)