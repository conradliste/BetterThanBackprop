# SmarterThanBackProp
Repository that contains experimental code on how to train neural networks better than normal backpropagation.

# Code Organization

The code is organized into 3 directories

* smartLearn: Code implementing different learning algorithms
* models: Neural network architectures
* dataLoaders: Code for preprocessing and loading in datasets
* examples: Executable code testing a learing algorithm on a specific dataset or game

# How to Run the Code in Developer Mode
Before running the code, first activate your virtual environment and run the `setup.py` file.

If you have a pip-based environment (e.g. virtualenv), run the command

`pip install --verbose -e .`

This will ensure that when you edit any package file and rerun the code, the executed code reflects your changes.

If you have a conda environment also add the `--no-deps` and `--no-build-isolation` flags.

`pip install --no-deps --no-build-isolation --verbose -e .`

With these additional flags, it forces you to install all dependencies manually. Although your build may not break if you do not use these flags and install the dependencies through pip, conda and pip handle dependencies independently and you could run into package conflicts.
