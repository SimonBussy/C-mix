## Welcome to C-mix : a high dimensional mixture model for censored durations

The code provided enable you to run both C-mix and CURE models in high dimension.
You may be interested if you face a supervised problem with temporal labels and you want to predict relative risks.

## Installation 

You must have :

- python >= 2.7

In order to install you must have the required Python dependencies:

    pip install -r requirements.txt

In order to install you must run

    python setup.py build_ext --inplace

In order to declare the package path to python you should put in the .bashrc file something like

    export PYTHONPATH=$PYTHONPATH:~/Programmation/Python/?

### Unittest

The library can be tested simply by running

    python -m unittest discover -v . "*_test.py"

in terminal. This shall check that everything is working and in order...

### Other files

You should definitely try the notebook "C-mix tutorial". 
It gives very useful example of how to use the model based on simulated data.
It will be very simple then to adapt it to your own data.
