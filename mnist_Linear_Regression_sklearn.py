# this .py file demonstrates the simplest implementation of linear regression on MNIST dataset

# Importing important libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Loadig the train.csv and test.csv file into python variable
train_x_orig = pd.read_csv('train.csv')
test_x_orig = pd.read_csv('test.csv')

# seperating labels from Train dataset
labels_train = (train_x_orig['label'].values).reshape(-1,1)

# Isolating digits from Train dataset
pixels_train = train_x_orig.drop('label', axis = 1)

# creating training and validation dataset by spliting the train.csv file in 70/30 ratio
from sklearn.model_selection import train_test_split

pix_train, pix_valid, label_train, label_valid = train_test_split(pixels_train, labels_train, test_size = 0.30)

# Converting data into array format
pix_train = (pix_train.values).astype('float32')
pix_test = (test_x_orig.values).astype('float32')
pix_valid = (pix_valid.values).astype('float32')


# normalizing brightness values of digits
pix_train /= 255.0
pix_test  /= 255.0
pix_valid /= 255.0

# extracting unique classes for clasification
num_classes = len(np.unique(labels_train))

# Importing libraries for Linear regression from sklearn package
from sklearn.linear_model import LinearRegression

regressor = LinearRegression(n_jobs = -1)

# fitting the regressor to the training data
regressor.fit(pix_train,label_train)

# Our regressor is now accustomed to the training data, lets see the predictions made by it 
y_pred_test = regressor.predict(pix_test)
y_pred_valid = regressor.predict(pix_valid)
y_pred_train = regressor.predict(pix_train)

# rounding of prediction to the nearest 'int' values, as we have 'int' values as classes
y_pred_test_labels = np.around(y_pred_test)
y_pred_valid_labels = np.around(y_pred_valid)
y_pred_train_labels = np.around(y_pred_train)

y_pred_test_labels = y_pred_test_labels.astype('int')
y_pred_valid_labels = y_pred_valid_labels.astype('int')
y_pred_train_labels - y_pred_train_labels.astype('int')

# defining a custom fucntion to check the prediction made by out regrssor against the actual label data

def count_per(y_pred_valid_labels, label_valid):
    count = 0
    for i in range(0,len(y_pred_valid_labels)):
        
        if y_pred_valid_labels[i,0] ==  label_valid[i,0]:
            count = count + 1
            
    return count
        
# calculating % match for validation dataset as well as training dataset
count1 = count_per(y_pred_valid_labels, label_valid)
valid_result_per = (count1/len(y_pred_valid)) * 100

count2 = count_per(y_pred_train_labels, labels_train)
train_result_per = (count2 / len(y_pred_train)) * 100


    