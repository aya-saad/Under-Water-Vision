import matplotlib.pyplot as plt
import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D, Activation
import numpy as np
from keras.datasets.mnist import load_data
from keras.preprocessing.image import ImageDataGenerator

class Alex_net:
         
     def __init__(self):
         self.name = "AlexNet"
         
     def preprocessing(self):
         (x_train1, y_train1), (x_test1, y_test1) = load_data()
         x_train =x_train1[:300]
         x_test = x_test1[:50]
         y_train =y_train1[:300]
         y_test = y_test1[:50]
         x_train.shape

        # Reshaping the array to 4-dims so that it can work with the Keras API
         x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
         x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

         train_datagen =ImageDataGenerator(rotation_range=10,rescale=1./255,shear_range=0.1,zoom_range=0.1,width_shift_range=0.1,height_shift_range=0.1)

         train_datagen.fit(x_train)
         train_generator = train_datagen.flow(x_train,y_train,batch_size=10)

         validation_datagen = ImageDataGenerator(rescale=1./255)
         train_datagen.fit(x_test)
         validation_generator = validation_datagen.flow(x_test,y_test)
         print('Number of images in x_train', x_train.shape[0])
         print('Number of images in x_test', x_test.shape[0])

    # def model(self):
         model = Sequential()
        # 1st Convolutional Layer

         model.add(Conv2D(96,(11,11), input_shape=(28,28,1), padding='same',strides=(4,4)))
         model.add(Activation('relu'))
        # Max Pooling
         model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same'))

        # 2nd Convolutional Layer
         model.add(Conv2D(filters=256, kernel_size=(11,11), strides=(1,1), padding='same'))
         model.add(Activation('relu'))
        # Max Pooling
         model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same'))

        # 3rd Convolutional Layer
         model.add(Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding='same'))
         model.add(Activation('relu'))

        # 4th Convolutional Layer
         model.add(Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding='same'))
         model.add(Activation('relu'))

        # 5th Convolutional Layer
         model.add(Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding='same'))
         model.add(Activation('relu'))
        # Max Pooling
         model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same'))

        # Passing it to a Fully Connected layer
         model.add(Flatten())
        # 1st Fully Connected Layer
         model.add(Dense(4096, input_shape=(28,28,1)))
         model.add(Activation('relu'))
        # Add Dropout to prevent overfitting
         model.add(Dropout(0.4))

        # 2nd Fully Connected Layer
         model.add(Dense(4096))
         model.add(Activation('relu'))
        # Add Dropout
         model.add(Dropout(0.4))

        # 3rd Fully Connected Layer
         model.add(Dense(1000))
         model.add(Activation('relu'))
        # Add Dropout
         model.add(Dropout(0.4))
        # Output Layer
         model.add(Dense(10))
         model.add(Activation('softmax'))



         model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
         model.fit_generator(train_generator,steps_per_epoch=30,epochs=10)


a=Alex_net()
a.preprocessing()

         

