## Welcome to C-mix : a high dimensional mixture model for censored durations

The code provided enable you to run both C-mix and CURE models in high dimension.
You may be interested if you face a supervised problem with temporal labels and you want to predict relative risks.

## Installation 

You must have :

- python >= 3.6
- R >= 3.3

In order to install you must have the required Python dependencies:

    pip install -r requirements.txt

### Unittest

The library can be tested simply by running

    python -m unittest discover -v . "*tests.py"

in terminal. This shall check that everything is working and in order...

To use the package outside the build directory, the build path should be added to the `PYTHONPATH` environment variable, as such (replace `$PWD` with the full path to the build directory if necessary):

    export PYTHONPATH=$PYTHONPATH:$PWD

For a permanent installation, this should be put in your shell setup script. To do so, you can run this from the _tick_ directory:

    echo 'export PYTHONPATH=$PYTHONPATH:'$PWD >> ~/.bashrc

Replace `.bashrc` with the variant for your shell (e.g. `.tcshrc`, `.zshrc`, `.cshrc` etc.).

### Other files

You should definitely try the notebook "C-mix tutorial". 
It gives very useful example of how to use the model based on simulated data.
It will be very simple then to adapt it to your own data.
