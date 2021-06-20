# Magnesita

## Getting started

### Pre-requisistes:

- python > 3.9
- sklearn
- pandas
- numpy
- jupyterlab
- tensorflow>2
- argparse
- silence-tensorflow

### Installation

I recommend using a virtual enviroment manager, such as (Anaconda)[https://www.anaconda.com/] for installing the dependecies.

For conda environments, use:

```
conda create -n magnesita python=3.9 pip
```

Then activate your enviroment using:

```
conda activate magnesita
```

Finally, install the dependecies using:

```
pip install -r requirements.txt
```

## Training a model

Use the *train.py* script for training your model. For example:

```
python train.py -pr data/md_raw_dataset.csv -pt data/md_target_dataset.csv -ps ./saves -v
```

All models and data requires for then evaluate the training will be stored in the folder defined in the *-ps* argument. The split between training and testing data is also saved in this folder for posterior use in the test step.

Example of output for training with *n_split=2* for the KFold algorithm:

```
Arguments:  Namespace(batch_size=64, epochs=100, verbose=True, n_splits=2, path_raw_data='data/md_raw_dataset.csv', path_target_data='data/md_target_dataset.csv', path_save='./saves', split_train_test=0.1)
Making data pipeline...
Training model...

Iteration 1/2
R2 score:  0.7614790537073751
Elapsed time:  12.551855325698853 sec
---------------

Iteration 2/2
R2 score:  0.7537238191999605
Elapsed time:  11.642426252365112 sec
---------------
Mean R2 score:  0.7576014364536678
```

## Testing a model

For testing the model, use the script *test.py*. For example:

```
python test.py -pr saves/test_raw.csv -pt saves/test_target.csv -ps ./saves
```

You can specify or not a path for the targets. If you do not specify, there will be no evaluation of the model, and only a file *predicted.csv* will be created with the predicted values for the input data.

You should use the same format csv data as provided in the training step for testing the model! Otherwise, it is not guaranteed good results.

Example of output for testing the model with target file:
```
Loading models...
Predicting data...

R2 score:  0.7590972360850207
Max error:  1.4622492066952015
Mean absolute error:  0.12487746792499992
Mean squared error:  0.04541922464909084
```