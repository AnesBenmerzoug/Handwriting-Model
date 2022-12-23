# Handwriting Model
Implementation of a model for handwriting synthesis using Long Short-Term Memory recurrent neural networks in PyTorch. Based on the work of Alex Graves described in this article: https://arxiv.org/abs/1308.0850

# Dataset
The dataset used to train this neural network is the [IAM On-Line Handwriting Database](http://www.fki.inf.unibe.ch/databases/iam-on-line-handwriting-database).
In order to train this network you have to register and download the following files:

* [lineStrokes-all.tar.gz](http://www.fki.inf.unibe.ch/DBs/iamOnDB/data/lineStrokes-all.tar.gz)
* [ascii-all.tar.gz](http://www.fki.inf.unibe.ch/DBs/iamOnDB/data/ascii-all.tar.gz)

## Usage

Create virtual environment and install packages:

```bash
python -m venv venv
source venv/bin/activate
pip install -e .
```

See the main script's help text for more information:

```bash
python -m handwriting_generator --help
```
