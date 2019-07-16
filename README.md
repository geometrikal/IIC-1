# IIC
My TensorFlow Implementation of https://arxiv.org/abs/1807.06653

Currently supports unsupervised clustering of MNIST data. More to come (I hope).

## Requirements

Tested on Python 3.6.8 and TensorFlow 1.14 with GPU acceleration.
I always recommend making a virtual environment.
To install required packages on a GPU system use:
```
pip install -r requirements.txt
```
For CPU systems replace `tensorflow-gpu==1.14.0` with `tensorflow==1.14.0` in `requirements.txt` before using pip.
Warning: I have not tried this.

## Repository Overview
* `data.py` contains code to construct a TensorFlow data pipeline where input perturbations are handled by the CPU.
* `graphs.py` contains code to construct various computational graphs, whose output connects to an IIC head.
* `models_iic.py` contains the `ClusterIIC` class, which implements unsupervised clustering.
* `utils.py` contains some utility functions.

## Running the Code
```
python models_iic.py
```
This will train IIC for the unsupervised clustering task using the MNIST data set.
I did my best to vigilantly adhere to the configuration in the original author's pyTorch code.
Running this code will print to console and produce a dynamically updated learning curve.

## Notes
The MNIST accuracy reported in https://arxiv.org/abs/1807.06653 suggests that all 5 subheads converge to ~99% accuracy.
My run-to-run variability suggests that while at least one head always converges to very high accuracy, some heads may
not attain ~99%. Often I see 2-4 subheads at ~99% with the other subheads at ~88% accuracy. I have not yet run the code
for the full 3200 epochs, so perhaps that resolves it.