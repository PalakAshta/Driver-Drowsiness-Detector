import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf
from PIL import Image
import os
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout

data = []
labels = []
classes = 4 
cur_path = os.getcwd()
classpath = os.path.join(cur_path,'data','train')
print(classpath)
yo = os.listdir(classpath)
#Retrieving the images and their labels 
for i in yo:
    print(i)
    path = os.path.join(classpath,i)
    images = os.listdir(path)

    for a in images:
        try:
            img2 = Image.open(path + '\\'+ a)
            #print('1')
            img2 = img2.resize((24,24))
            image = img2.convert('L')
            #print('2')
            image = np.array(image)
            #print('3')
            #sim = Image.fromarray(image)
            data.append(image)
            #print('4')
            if(i=="Open"):
                labels.append(0)
            elif(i=="Closed"):
                labels.append(1)
            elif(i=="yawn"):
                labels.append(2)
            else:
                labels.append(3)
            #print('5')
        except:
            print("Error loading image")

#Converting lists into numpy arrays
data = np.array(data)
labels = np.array(labels)
print(data.shape, labels.shape)
#Splitting training and testing dataset
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.12, random_state=42)

print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

# #Converting the labels into one hot encoding
y_train = to_categorical(y_train, 4)
y_test = to_categorical(y_test, 4)
print(y_train)
X_train = X_train.reshape(-1, 24, 24, 1)
#Building the model
model = Sequential()
model.add(Conv2D(filters=32, kernel_size=(5,5), activation='relu', input_shape=X_train.shape[1:]))
model.add(Conv2D(filters=32, kernel_size=(5,5), activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(rate=0.25))
model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(rate=0.25))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(rate=0.5))
model.add(Dense(4, activation='softmax'))

#Compilation of the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

epochs = 15
history = model.fit(X_train, y_train, batch_size=32, epochs=epochs, validation_data=(X_test, y_test))
model.save("cnnCat2.h5")

#plotting graphs for accuracy 
plt.figure(0)
plt.plot(history.history['accuracy'], label='training accuracy')
plt.plot(history.history['val_accuracy'], label='val accuracy')
plt.title('Accuracy')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.legend()
plt.show()
