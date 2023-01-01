# Handwriting Model
Implementation of a model for handwriting synthesis using
a Long Short-Term Memory recurrent neural networks in PyTorch.

Based on the work of Alex Graves described in this article: https://arxiv.org/abs/1308.0850

# Dataset
The dataset used to train this neural network is the [IAM On-Line Handwriting Database](http://www.fki.inf.unibe.ch/databases/iam-on-line-handwriting-database).
In order to train this network you have to register then download the following files:

* [lineStrokes-all.tar.gz](http://www.fki.inf.unibe.ch/DBs/iamOnDB/data/lineStrokes-all.tar.gz)
* [ascii-all.tar.gz](http://www.fki.inf.unibe.ch/DBs/iamOnDB/data/ascii-all.tar.gz)

Put both of inside the [data](data) directory.

## Usage

Create virtual environment and install packages:

```shell
python -m venv venv
source venv/bin/activate
pip install -e .
```

Then execute the following to train the model:

```shell
python -m handwriting_generator
```

See the main script's help text for more information:

```shell
python -m handwriting_generator --help
```
