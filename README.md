# SmarterThanBackProp
Repository that contains details on how to train neural networks better than normal backpropagation

# How to Run the Code in Edit Mode
Before running the code, first activate your virtual environment and run the `setup.py` file.

If you have a pip-based environment (e.g. virtualenv), run the command

`pip install --editable`

This will ensure that when you edit the package, the code will change.

If you have a conda environment also add the `--no-deps` and `--no-build-isolation` flags.

`pip install --no-deps --no-build-isolation --editable`

With these flags, it forces you to install all dependencies manually.
Although your build may not break if you do not use these flags and install the dependencies through pip, conda and pip handle dependencies independently and you could run into conflicts.