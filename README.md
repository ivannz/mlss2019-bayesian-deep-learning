# MLSS2019: Bayesian Deep Learning

## Installation: colab

In Google colab there is no need to clone the repo or preinstall anything --
all jupyter runtimes come with the basic packages like numpy, scipy, and
matplotlib and deep learning libraries keras, tensorflow, and pytorch.

The only step to make is to change the runtime type to GPU in 
**Edit > Notebook settings or Runtime>Change runtime type** by selecting
**GPU as Hardware accelerator**.


## Installation: local install

Please make sure that you have the following packages installed:
* tqdm
* numpy
* torch >= 1.1

The most convenient way to ensure this is use Anaconda with python 3.7.

When all prerequisites have been met, please, clone this repository and
install it with:

```bash
git clone https://github.com/ivannz/mlss2019-bayesian-deep-learning.git

cd mlss2019-bayesian-deep-learning

pip install --editable .
```

This will install the necessary service python code that will make the seminar
much easier and your learning experience better.
