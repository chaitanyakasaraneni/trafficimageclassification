#!/usr/bin/env python
# coding: utf-8

# # CMPE 255 - Programming Assignment 2: Image Classification

# ### Importiing the Required libraries

# In[1]:


import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import to_categorical
from keras.preprocessing import image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from tqdm import tqdm
import os


# ### Preprocessing (kind of)
# 
# Storing the Labels into a list for future usage & reference.
# Storing the list of image names into a CSV file and loading that into "train" dataframe for future reference

# In[2]:


labels = []
with open("traffic/traffic/train.labels", "r") as fh:
    train_lines = fh.readlines() 
for line in train_lines:
    splitline = line.split('\n')
    labels.append(splitline[0])
# labels
import os, csv
# f=open("traffic/traffic/contents.csv",'w')
# w=csv.writer(f)
# w.writerow(["id"])
# for path, dirs, files in os.walk("traffic/traffic/train"):
#     for filename in files:
#         w.writerow([filename])
files_list = os.listdir("traffic/traffic/train")
resultfile = open("traffic/traffic/trail2.csv",'w')
resultfile.write('id' + "\n")
for r in files_list:
    resultfile.write(r + "\n")
resultfile.close()
train = pd.read_csv('traffic/traffic/trail2.csv')
train.shape[0]


# In this step, we will read all the training images, store them in a list by using tensorflow's img_to_array() function, and finally convert that list into a numpy array.

# In[3]:


train_image = []
for i in tqdm(range(train.shape[0])):
    img = image.load_img('traffic/traffic/train/'+train['id'][i], target_size = (28,28,1), color_mode = "grayscale")
    img = image.img_to_array(img)
    img = img/255
    train_image.append(img)
X = np.array(train_image)


# As it is a multi-class classification problem (14 classes), we will one-hot encode the target variable.

# In[4]:


y = labels
y = to_categorical(y)
len(X) , X


# Creating a validation set from the training data

# In[5]:


X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2)


# ### Defining the model structure.
# 
# We will create a simple architecture with 2 convolutional layers, one dense hidden layer and an output layer.

# In[6]:


# from sklearn.preprocessing import StandardScaler  
# scaler = StandardScaler()  
# scaler.fit(X_train)

# X_train = scaler.transform(X_train)  
# X_test = scaler.transform(X_test)
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),activation='relu',input_shape=(28,28,1)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(15, activation='softmax'))


# Next, we will compile the model we’ve created.

# In[7]:


model.compile(loss='categorical_crossentropy',optimizer='Adam',metrics=['accuracy'])


# ### Training the model.
# 
# In this step, we will train the model on the training set images and validate it using, you guessed it, the validation set.

# In[8]:


model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))


# ### Making predictions!
# 
# We’ll initially follow the steps we performed when dealing with the training data. Load the test images and predict their classes using the model.predict_classes() function.

# In[9]:


files_list = os.listdir("traffic/traffic/test")
resultfile = open("traffic/traffic/test.csv",'w')
resultfile.write('id' + "\n")
for r in files_list:
    resultfile.write(r + "\n")
resultfile.close()
test = pd.read_csv('traffic/traffic/test.csv')
test.shape[0]


# In[10]:


test_image = []
for i in tqdm(range(test.shape[0])):
    img = image.load_img('traffic/traffic/test/'+train['id'][i], target_size = (28,28,1), color_mode = "grayscale")
    img = image.img_to_array(img)
    img = img/255
    test_image.append(img)
test = np.array(test_image)


# In[11]:


predictions = model.predict_classes(test)


# ### Writing the predicted classses into a file

# In[12]:


with open('predictions.dat', 'w') as f:
        for cls in predictions:
            f.write("%s\n" % cls)


# ### References:
# 
# https://www.analyticsvidhya.com/blog/2019/01/build-image-classification-model-10-minutes/
# 
