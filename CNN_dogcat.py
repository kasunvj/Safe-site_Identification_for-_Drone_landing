import os
import numpy as np
import pandas as pd
import scipy
import sklearn
import keras
from keras.models import Sequential
import cv2
from skimage import io
#%matplotlib inline

#defining file path
cats = os.listdir("D:/Kasun Projects/database/dogscats/dogscats/train/cats")
dogs = os.listdir("D:/Kasun Projects/database/dogscats/dogscats/train/dogs")
filepath  = "D:/Kasun Projects/database/dogscats/dogscats/train/cats/"
filepath2 = "D:/Kasun Projects/database/dogscats/dogscats/train/dogs/"

#loading images
images = []
lable = []

for i in cats:
	image = scipy.misc.imread(filepath + i)
	images.append(image)
	lable.append(0) #for cats

for i in dogs:
	image = scipy.misc.imread(filepath2 + i)
	images.append(image)
	lable.append(1) #for dogs

#resizing images 
for i in range(0,23000):
	images[i] = cv2.resize(images[i],(300,300)) 

#converting images to array (300,300,3)
images = np.array(images)
lable = np.array(lable)

# Defining the hyperparameters

filters =10
filtersize = (5,5)

epochs = 5
batchsize  = 128

input_shape = (300,300,3)

#converting target variable to the required size 

from keras.utils.np_utils import to_categorical
lable = to_categorical(lable)

#defining the model
model = Sequential()

model.add(keras.layers.InputLayer(input_shape = input_shape))

model.add(keras.layers.convolutional.Conv2D(filters, filtersize,strides =(1,1),padding ='valid',data_format = "channels_last",activation = 'relu'))
model.add(keras.layers.MaxPooling2D(pool_size = (2,2)))
model.add(keras.layers.Flatten())

model.add(keras.layers.Dense(units=2, input_dim=50, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(images, lable,epochs=epochs ,batch_size= batchsize,validation_split=0.3)

model.summary()

print ("Tadaa")