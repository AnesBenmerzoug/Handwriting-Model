# Handwriting Model
Implementation of a model for handwriting synthesis using Long Short-Term Memory recurrent neural networks in PyTorch. Based on the work of Alex Graves described in this article: https://arxiv.org/abs/1308.0850

# Dataset
The dataset used to train this neural network is the [IAM On-Line Handwriting Database](http://www.fki.inf.unibe.ch/databases/iam-on-line-handwriting-database). In order to train this network you have to register and download the following files:

* [data/lineStrokes-all.tar.gz](http://www.fki.inf.unibe.ch/DBs/iamOnDB/data/lineStrokes-all.tar.gz)
* [ascii-all.tar.gz](http://www.fki.inf.unibe.ch/DBs/iamOnDB/data/ascii-all.tar.gz)

And the directory structure has to be the following:
```
    .
    ├── main.py                
    ├── README.md
    ├── parameters.yaml       
    ├── LICENSE
    ├── trained_models          # Trained models
    ├── src                     # Source files
    └── data                    # Data files
        ├── ascii          # text files that contain the written text
        |   ├── a01
        |   |   ├── a01-000
        |   |   |   ├── a01-000u.txt
        |   |   |   └── a01-000x.txt
        |   |   └── ...
        |   └── ...
        └── lineStrokes    # xml files that contain the strokes
            ├── a01
            |   ├── a01-000
            |   |   ├── a01-000u-01.xml
            |   |   └── a01-000u-02.xml
            |   |   └── ...
            |   └── ...
            └── ...
    
```

## Usage

Create virtual environment and install packages:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

See the main script's help text for more information:

```bash
python main.py --help
```
