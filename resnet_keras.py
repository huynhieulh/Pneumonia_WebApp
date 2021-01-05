# -*- coding: utf-8 -*-
import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D
import numpy as np
import cv2
import os
#Split train / valid
from sklearn.model_selection import train_test_split as tts

#Size image
IMGS = 256
#Directory paths:
main = './chest_xray/'
categ = ['test', 'train', 'val']
classes = ['NORMAL','PNEUMONIA']
images = []
labels = []
for i in categ:
    sub_path = os.path.join(main, i)
    for j  in classes:
        path = os.path.join(sub_path, j) 
        temp = os.listdir(path)
        for x in temp:
            addr = os.path.join(path, x)
            img_arr = cv2.imread(addr)
            img_arr = cv2.resize(img_arr, (IMGS, IMGS))
            images.append(img_arr)
            if j == 'PNEUMONIA':
                l = 1
            else:
                l = 0
            labels.append(l)
print('Done.')

images = np.array(images)
labels = np.array(labels)
print(images.shape, labels.shape)

x_train, x_test, y_train, y_test = tts(images, labels, random_state = 42, test_size = .20)
x_train.shape, y_train.shape, x_test.shape, y_test.shape

x_train = x_train.reshape(-1, IMGS, IMGS, 3)
x_test = x_test.reshape(-1, IMGS, IMGS, 3)
print('Done.')
print(x_train.shape, x_test.shape)

model_9 = Sequential()

# 1st Convolution block
model_9.add(Conv2D(16, (3, 3), strides=1, padding='same', activation='relu',
                        input_shape=(256, 256, 3)))
model_9.add(MaxPooling2D((2, 2), strides=2, padding='same'))

# 2nd Convolution block
model_9.add(Conv2D(32, (1, 1), strides=1, padding='same', activation='relu'))
model_9.add(Conv2D(32, (3, 3), strides=1, padding='same', activation='relu'))
model_9.add(Conv2D(32, (1, 1), strides=1, padding='same', activation='relu'))
model_9.add(MaxPooling2D((2, 2), strides=2, padding='same'))

# 3rd Convolution block
model_9.add(Conv2D(64, (1, 1), strides=1, padding='same', activation='relu'))
model_9.add(Conv2D(64, (3, 3), strides=1, padding='same', activation='relu'))
model_9.add(Conv2D(64, (1, 1), strides=1, padding='same', activation='relu'))
model_9.add(MaxPooling2D((2, 2), strides=2))

# 4th Convolution block
model_9.add(Conv2D(96, (1, 1), strides=1, padding='same', activation='relu'))
model_9.add(Conv2D(96, (3, 3), strides=1, padding='same', activation='relu'))
model_9.add(Conv2D(96, (1, 1), strides=1, padding='same', activation='relu'))
model_9.add(MaxPooling2D((2, 2), strides=2))

# 5th Convolution block
model_9.add(Conv2D(128, (1, 1), strides=1, padding='same', activation='relu'))
model_9.add(Conv2D(128, (3, 3), strides=1, padding='same', activation='relu'))
model_9.add(Conv2D(128, (1, 1), strides=1, padding='same', activation='relu'))
model_9.add(MaxPooling2D((2, 2), strides=2))

# Flattened the layer
model_9.add(Flatten())

# Fully connected layers
model_9.add(Dense(64, activation='relu'))
model_9.add(Dropout(0.2))
model_9.add(Dense(1, activation='sigmoid'))

model_9.summary()

model_9.compile(optimizer= keras.optimizers.Adam(lr=0.0001), loss='binary_crossentropy', metrics=['accuracy'])
model_9.fit(x_train, y_train, epochs = 10, batch_size = 32, validation_data=(x_test, y_test))

print(model_9.evaluate(x_test, y_test))

y_pred = model_9.predict(x_test)

y_pred = np.round(y_pred)
y_pred = y_pred.reshape(-1,)
y_test = y_test.astype('float')
y_test.shape

from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test, y_pred))
print(classification_report(np.round(y_pred),y_test))