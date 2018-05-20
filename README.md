# Handwriting Model
Implementation of a model for handwriting synthesis using Long Short-Term Memory recurrent neural networks in PyTorch. Based on the work of Alex Graves described in this article: https://arxiv.org/abs/1308.0850

# Dataset
The dataset used to train this neural network is the [IAM On-Line Handwriting Database](http://www.fki.inf.unibe.ch/databases/iam-on-line-handwriting-database). In order to train this network you have to register and download the following files:

* [data/lineStrokes-all.tar.gz](http://www.fki.inf.unibe.ch/DBs/iamOnDB/data/lineStrokes-all.tar.gz)
* [ascii-all.tar.gz](http://www.fki.inf.unibe.ch/DBs/iamOnDB/data/ascii-all.tar.gz)
* [task1.tar.gz](http://www.fki.inf.unibe.ch/DBs/iamOnDB/task1.tar.gz)

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
        ├── lineStrokes    # xml files that contain the strokes
        |   ├── a01
        |   |   ├── a01-000
        |   |   |   ├── a01-000u-01.xml
        |   |   |   └── a01-000u-02.xml
        |   |   |   └── ...
        |   |   └── ...
        |   └── ...
        └── task1         # text files that contain a list of the samples to correspond to each set
            ├── trainset.txt
            ├── testset_v.txt
            ├── testset_t.txt
            ├── testset_f.txt
            └── ...
    
    
```

## Requierements
```
Python  3.6
PyTorch 0.3.1
numpy   1.14.1
```
