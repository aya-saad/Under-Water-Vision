import os
import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D, Activation,BatchNormalization
from keras import optimizers
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix,f1_score, precision_score, recall_score,classification_report



class Net:

    def __init__(self, name='AlexNet', input_width=150, input_height=150, input_channels=3, num_classes=2, learning_rate=0.01,
                 momentum=0.9, keep_prob=0.8,
                 #model_file='plankton-classifier.tfl'
                 ):
        self.name = name

        self.input_width = input_width
        self.input_height = input_height
        self.input_channels = input_channels

        self.num_classes = num_classes

        self.learning_rate = learning_rate

        self.momentum = momentum
        self.keep_prob = keep_prob

        self.random_mean = 0
        self.random_stddev = 0.01
        #self.check_point_file = model_file


    def pre_processing(train_dir,validation_dir,test_dir):
        # All images will be rescaled by 1./255
        train_datagen = ImageDataGenerator(featurewise_center=True,featurewise_std_normalization=True,
                                           rotation_range=25,horizontal_flip=True,vertical_flip=True,rescale=1./255)
        test_datagen = ImageDataGenerator(rescale=1./255)
        #target dir, images resized to 150*150, use binary cross entropy
        train_generator = train_datagen.flow_from_directory(train_dir,target_size=(150, 150),batch_size=30,class_mode='binary')
        validation_generator = test_datagen.flow_from_directory(validation_dir,target_size=(150, 150),batch_size=30,class_mode='binary')
        test_generator = test_datagen.flow_from_directory(test_dir,target_size=(150,150),batch_size=30,class_mode=None)
        return train_generator,test_generator,validation_generator

    def train(model,train_generator,validation_generator,epochs=10, batch_size=30):
        model.fit_generator(train_generator,steps_per_epoch=30,epochs=1,
                              validation_data=validation_generator,validation_steps=50)
        return model
        
    def evaluate(model,validation_generator,test_generator):
        STEP_SIZE_VALID=validation_generator.n//validation_generator.batch_size
        STEP_SIZE_TEST=test_generator.n//test_generator.batch_size
        model.evaluate_generator(validation_generator,steps=STEP_SIZE_VALID)
        test_generator.reset()
        predictions =model.predict_generator(test_generator,steps=STEP_SIZE_TEST,verbose=1)
        val_preds =np.argmax(predictions,axis=1)
        val_true = validation_generator.classes
        cm = confusion_matrix(val_true, val_preds)
        print(cm)
        print(classification_report(val_true, val_preds))

    
    def build_model(self):
     
        #print(self.name)
        #self.model_file = model_file
        #print(self.model_file)
        if self.name == 'OrgNet':
            return self.OrgNet_model()
        if self.name == 'LeNet':
            return self.LeNet_model()
        elif self.name == 'MINST':
            return self.MINST_model()
        #elif self.name == 'CIFAR10':
         #   return self.__build_CIFAR10()
        elif self.name == 'AlexNet':
            return self.AlexNet_model()
        elif self.name == 'VGGNet':
            return self.VGGNet_model()
        elif self.name == 'GoogLeNet':
            return self.GoogLeNet_model()
        elif self.name == 'ResNet':
            return self.ResNet_model()
        #elif self.name == 'ResNeXt':
         #   return self.__build_ResNeXt()
        #elif self.name == 'PlankNet':
         #   return self.__build_PlankNet()
        elif self.name == 'CoapNet':
            print(self.name)
            return self.CoapNet_model()

    def AlexNet_model(img_shape=(150,150,3),num_classes=2):
        #inputsize = self.input_width * self.input_height * self.input_channels
        #print("Inputlayer-size: %d" % (inputsize))
        alexnet = Sequential()
        # Layer 1
        alexnet.add(Conv2D(96, (11, 11), input_shape=img_shape,padding='same'))
	#alexnet.add(BatchNormalization())
        alexnet.add(Activation('relu'))
        alexnet.add(MaxPooling2D(pool_size=(2, 2)))

        # Layer 2
        alexnet.add(Conv2D(256, (5, 5), padding='same'))
        #alexnet.add(BatchNormalization())
        alexnet.add(Activation('relu'))
        alexnet.add(MaxPooling2D(pool_size=(2, 2)))

        # Layer 3
        #alexnet.add(ZeroPadding2D((1, 1)))
        alexnet.add(Conv2D(512, (3, 3), padding='same'))
        #alexnet.add(BatchNormalization())
        alexnet.add(Activation('relu'))
        alexnet.add(MaxPooling2D(pool_size=(2, 2)))

        # Layer 4
        #alexnet.add(ZeroPadding2D((1, 1)))
        alexnet.add(Conv2D(1024, (3, 3), padding='same'))
        #alexnet.add(BatchNormalization())
        alexnet.add(Activation('relu'))

        # Layer 5
        #alexnet.add(ZeroPadding2D((1, 1)))
        alexnet.add(Conv2D(1024, (3, 3), padding='same'))
        #alexnet.add(BatchNormalization())
        alexnet.add(Activation('relu'))
        alexnet.add(MaxPooling2D(pool_size=(2, 2)))

        # Layer 6
        alexnet.add(Flatten())
        alexnet.add(Dense(3072))
        #alexnet.add(BatchNormalization())
        alexnet.add(Activation('relu'))
        alexnet.add(Dropout(0.5))

        # Layer 7
        alexnet.add(Dense(4096))
        #alexnet.add(BatchNormalization())
        alexnet.add(Activation('relu'))
        alexnet.add(Dropout(0.5))

        # Layer 8
        alexnet.add(Dense(num_classes))
        #alexnet.add(BatchNormalization())
        alexnet.add(Activation('softmax'))
        #adam = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
        #train_generator = zip(pre_generator, aug_generator)
        alexnet.compile(loss='sparse_categorical_crossentropy',optimizer=optimizers.RMSprop(lr=1e-4),metrics=['acc'])
        return alexnet

    def LeNet_model(img_shape=(150,150,3),num_classes=2):
        LeNet = Sequential()
        reg=keras.regularizers.l2(0.01)
        LeNet.add(Conv2D(6,(5,5),activation='relu',input_shape=img_shape,
                         Regularizers=reg))
        LeNet.add(MaxPooling2D(pool_size=(2, 2)))
        LeNet.add(Conv2D(16,(5,5),activation='relu',input_shape=img_shape,
                         Regularizers=reg))
        LeNet.add(MaxPooling2D(pool_size=(2, 2)))
        LeNet.add(Flatten())
        LeNet.add(Dense(120))
        LeNet.add(ASctivation('tanh'))
        LeNet.add(Dropout(0.8))
        LeNet.add(Dense(84))
        LeNet.add(Activation('tanh'))
        LeNet.add(Dense(num_classes))
        LeNet.add(Activation('softmax'))
        LeNet.compile(loss='sparse_categorical_crossentropy',
                      optimizer=optimizers.adam(lr=0.01),metrics=['acc'])
        return LeNet

    def MNIST_model(img_shape=(150,150,3),num_classes=2):
        MNIST = Sequential()
        reg=keras.regularizers.l2(0.01)
        MNIST.add(Conv2D(32,(3,3),activation='relu',input_shape=img_shape,
                         regularizers=reg))
        MNIST.add(MaxPooling2D(pool_size=(2, 2)))
        MNIST.add(BatchNormalization())
        MNIST.add(Flatten())
        MNIST.add(Dense(128))
        MNIST.add(activation('tanh'))
        MNIST.add(Dropout(0.8))
        MNIST.add(Dense(256))
        MNIST.add(Activation('tanh'))
        MNIST.add(Dense(num_classes))
        MNIST.add(Activation('softmax'))
        MNIST.compile(loss='sparse_categorical_crossentropy',
                      optimizer=optimizers.adam(lr=0.01),metrics=['acc'])
        return MNIST


    def OrgNet_model(img_shape=(150,150,3),num_classes=2):
        OrgNet = Sequential()
        OrgNet.add(Conv2D(32,(3,3),activation='relu',input_shape=img_shape))
        OrgNet.add(MaxPooling2D(pool_size=(2, 2)))
        OrgNet.add(Conv2D(32,(3,3),activation='relu'))
        OrgNet.add(Conv2D(32,(3,3),activation='relu'))
        OrgNet.add(Conv2D(32,(3,3),activation='relu'))
        OrgNet.add(Conv2D(32,(3,3),activation='relu'))
        OrgNet.add(Conv2D(32,(3,3),activation='relu'))
        OrgNet.add(MaxPooling2D(pool_size=(2, 2)))
        OrgNet.add(Flatten())
        OrgNet.add(Dense(512))
        OrgNet.add(Activation('relu'))
        OrgNet.add(Dense(num_classes))
        OrgNet.add(Activation('softmax'))
        OrgNet.compile(loss='sparse_categorical_crossentropy',
                      optimizer=optimizers.adam(lr=0.01),metrics=['acc'])
        return OrgNet

    def CoapNet_model(img_shape=(150,150,3),num_classes=2):
        CoapNet = Sequential()
        CoapNet.add(Conv2D(64,(3,3),activation='relu',input_shape=img_shape))
        CoapNet.add(MaxPooling2D(pool_size=(2, 2)))
        CoapNet.add(Conv2D(128,(3,3),activation='relu'))
        CoapNet.add(MaxPooling2D(pool_size=(2, 2)))
        CoapNet.add(Conv2D(256,(3,3),activation='relu'))
        CoapNet.add(MaxPooling2D(pool_size=(2, 2)))
        CoapNet.add(Conv2D(512,(3,3),activation='relu'))
        CoapNet.add(MaxPooling2D(pool_size=(2, 2)))
        CoapNet.add(Flatten())
        CoapNet.add(Dense(512))
        CoapNet.add(Activation('relu'))
        CoapNet.add(Dense(256))
        CoapNet.add(Activation('relu'))
        CoapNet.add(Dense(256))
        CoapNet.add(Activation('relu'))
        CoapNet.add(Dense(num_classes))
        CoapNet.add(Activation('softmax'))
        CoapNet.compile(loss='sparse_categorical_crossentropy',
                      optimizer=optimizers.adam(lr=0.01),metrics=['acc'])
        return CoapNet

    def CIFAR10_model(img_shape=(150,150,3),num_classes=2):
        CIFAR10 = Sequential()
        CIFAR10.add(Conv2D(32,(3,3),activation='relu',input_shape=img_shape))
        CIFAR10.add(MaxPooling2D(pool_size=(2, 2)))
        CIFAR10.add(Conv2D(64,(3,3),activation='relu'))
        CIFAR10.add(Conv2D(64,(3,3),activation='relu'))
        CIFAR10.add(MaxPooling2D(pool_size=(2, 2)))
        CIFAR10.add(Flatten())
        CIFAR10.add(Dense(512))
        CIFAR10.add(Activation('relu'))
        CIFAR10.add(Dropout(0.5))
        CIFAR10.add(Dense(num_classes))
        CIFAR10.add(Activation('softmax'))
        CIFAR10.compile(loss='sparse_categorical_crossentropy',
                      optimizer=optimizers.adam(lr=0.01),metrics=['acc'])
        return CIFAR10

    def PlankNet_model(img_shape=(150,150,3),num_classes=2):
        PlankNet = Sequential()
        PlankNet.add(Conv2D(96,(13,13),strides=4,activation='relu',input_shape=img_shape))
        PlankNet.add(MaxPooling2D(pool_size=(2, 2)))
        PlankNet.add(BatchNormalization())
        PlankNet.add(Conv2D(256,(7,7),activation='relu',padding='same'))
        PlankNet.add(MaxPooling2D(pool_size=(2, 2)))
        PlankNet.add(Conv2D(512,(3,3),activation='relu'))
        PlankNet.add(Conv2D(512,(3,3),activation='relu'))
        PlankNet.add(Conv2D(512,(3,3),activation='relu'))
        PlankNet.add(MaxPooling2D(pool_size=(2, 2)))
        PlankNet.add(Conv2D(512,(3,3),activation='relu'))
        PlankNet.add(Conv2D(512,(3,3),activation='relu'))
        PlankNet.add(Conv2D(512,(3,3),activation='relu'))
        PlankNet.add(MaxPooling2D(pool_size=(2, 2)))
        PlankNet.add(BatchNormalization())
        PlankNet.add(Flatten())
        PlankNet.add(Dense(4096, activation ='tanh'))
        PlankNet.add(Dropout(0.5))
        PlankNet.add(Dense(4096, activation ='tanh'))
        PlankNet.add(Dropout(0.5))
        PlankNet.Dense((num_classes))
        PlankNet.add(Activation('softmax'))
        PlankNet.compile(loss='sparse_categorical_crossentropy',
                      optimizer=optimizers.adam(lr=0.01),metrics=['acc'])
        return PlankNet

    


        

        
                  
        
        
        
        

        
    
