# ALL-for-MNIST

This REPO has been created for testing following models against the MNIST dataset.
1. Convolutional Neural Network using keras
2. Linear Regression using sklearn

Both the implementations are the basic structure of how to load data, preprocess it, create architecture and obtain results. MNIST data can been downloaded from the Kaggle [MNIST challenge](https://www.kaggle.com/deepnetwork/simple-cnn-for-mnist-dataset).

## Getting Started
### Prerequisites

* NVIDIA GPU (TF-GPU) will reduce the time multiple folds taken during training(not compulsory).
* Libraries required: pandas, numpy, matplotlib, keras(TF), sklearn.

### Content

This Repository contains the folowing files:
* mnist_CNN_keras.py 
* mnist_CNN_keras_notebook.ipynb
* mnist_Linear_Regression_sklearn.py
* mnist_Linear_Regression_sklearn_notebook.pynb
* test.csv
* train.csv
* mnist_CNN_FlowChart.jpg

### Description

For each method we have 2 files, one is simple .py file without any visualization and another is .ipynb notebook with details and visualization for the same.
"mnist_CNN_FlowChart.jpg" contains the basic understanding of how data(image) is processed and fed into the CNN which is user defined.

![mnist_CNN_FlowChart](https://user-images.githubusercontent.com/47072039/89782756-7de20180-db33-11ea-9c85-f508e882cea2.jpg)

## Acknowledgments

* Inspiration- [sentdex](https://pythonprogramming.net/)
