# this is a simple guide to build the most basic CNN using keras and perform classification in MNIST dataset
# importing important libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# loadig the train.csv and test.csv file into python variable
train_x_orig = pd.read_csv('train.csv')
test_x_orig = pd.read_csv('test.csv')

# training dataset comes with digits and its labels 
# shape of train_x_orig is (42000,785), and a digit is of size (28x28 = 784)
# we will be seperating the label column and storing it in a different variable
labels_train = (train_x_orig['label'].values).reshape(-1,1)

# since the label column has its own variable now, we will also keep the digits in its own variable
pixels_train = train_x_orig.drop('label', axis = 1)

# importing train_test_split function to divide the data-set into 70% train and 30% test set 
from sklearn.model_selection import train_test_split

# spliting the train data-set(digits,lables) into training and validation set with 70/30 ratio 
pix_train, pix_valid, label_train, label_valid = train_test_split(pixels_train, labels_train, test_size = 0.30)

# our dataset is still in data frame format, we will be converting it into array
pix_train = (pix_train.values).astype('float32')
pix_test = (test_x_orig.values).astype('float32')
pix_valid = (pix_valid.values).astype('float32')


# digits are images of dim 28,28 and have brightness values ranging from 0-255
# we will be normalizing this so that our results are obtained much faster
pix_train /= 255.0
pix_test  /= 255.0
pix_valid /= 255.0

# CNN is responsible to extract features from the image , so we need to convert 784 numbers to 28x28
pix_train = pix_train.reshape(pix_train.shape[0], 28,28,1)
pix_test = pix_test.reshape(pix_test.shape[0], 28, 28,1)
pix_valid = pix_valid.reshape(pix_valid.shape[0], 28, 28,1)

# determining the number of unique digits i.e classes we have to classify from 
num_classes = len(np.unique(labels_train))

# converting the labels to categorical format which is much easier for the model to understand
from keras.utils import to_categorical

label_train = to_categorical(label_train)
label_valid = to_categorical(label_valid)

# importing CNN fucntions from keras 
from keras.layers import Dense, Conv2D, Flatten, Dropout, BatchNormalization, MaxPooling2D
from keras.models import Sequential

# defining model parameters and its architecture
model = Sequential()

model.add(Conv2D(filters = 64, kernel_size = (3,3), strides = 1, padding = 'valid', activation = 'relu', input_shape = pix_train.shape[1:] ))
model.add(Conv2D(filters = 64, kernel_size = (3,3), strides = 1, padding = 'valid', activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2,2)))

model.add(Conv2D(filters = 128, kernel_size = (3,3), strides = 1, padding = 'valid', activation = 'relu'))
model.add(Conv2D(filters = 128, kernel_size = (3,3), strides = 1, padding = 'valid', activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2,2)))

model.add(Flatten())

model.add(Dense(2048, activation = 'relu'))
model.add(Dropout(0.4))
model.add(BatchNormalization())

model.add(Dense(512, activation = 'relu'))
model.add(Dropout(0.3))
model.add(BatchNormalization())

model.add(Dense(256, activation = 'relu'))
model.add(Dropout(0.3))
model.add(BatchNormalization())

model.add(Dense(num_classes, activation = 'softmax'))

model.compile(loss = 'categorical_crossentropy', optimizer = 'SGD', metrics = ['accuracy'])

# feeding in data to the model architecture created, along with validation dataset
hist = model.fit(pix_train, label_train, epochs = 5, batch_size = 64, validation_data = (pix_valid, label_valid))


# plot loss during training
plt.subplot(211)
plt.title('Loss')
plt.plot(hist.history['loss'], label='train')
plt.plot(hist.history['val_loss'], label='test')
plt.legend()
# plot accuracy during training
plt.subplot(212)
plt.title('Accuracy')
plt.plot(hist.history['accuracy'], label='train')
plt.plot(hist.history['val_accuracy'], label='test')
plt.legend()
plt.show()

# obtaining results from the model by giving it totaly new dataset (test_dataset)
results = model.predict(pix_test)

results = np.argmax(results, axis = 1)

results = pd.Series(results, name = 'Label')

sub = pd.concat([pd.Series(range(1,28001), name = 'ImageId'), results], axis = 1)

# saving results to CSV file 
#sub.to_csv('test_dataset_results.csv', index = False)
