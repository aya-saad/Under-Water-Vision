import os
import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D, Activation,BatchNormalization,ZeroPadding2D
from keras import optimizers
from keras.regularizers import l2
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix,f1_score, precision_score, recall_score
from sklearn import metrics
from keras.regularizers import l2
from keras.applications.resnet50 import ResNet50
from pool import PoolHelper,LRN
class Net:

    def __init__(self,train_dir,validation_dir,test_dir, name='OrgNet',input_width=150, input_height=150, input_channels=3, num_classes=2, learning_rate=0.001,
                 momentum=0.9, keep_prob=0.8, model_file = 'cat-dog-classifier.tfl'):
                 
        self.name = name
        self.train_dir = train_dir
        self.test_dir= test_dir
        self.validation_dir = validation_dir
        self.input_width = input_width
        self.input_height = input_height
        self.input_channels = input_channels

        self.num_classes = num_classes

        self.learning_rate = learning_rate

        self.momentum = momentum
        self.keep_prob = keep_prob

        self.random_mean = 0
        self.random_stddev = 0.01
        print("Net init")
        #self.pre_processing()
        #self.build_model(self.name)
        #self.check_pointfile = model_file


    def pre_processing(train_dir,validation_dir,test_dir):
        # All images will be rescaled by 1./255
        train_datagen = ImageDataGenerator(featurewise_center=True,featurewise_std_normalization=True,
                                           rotation_range=25,horizontal_flip=True,vertical_flip=True,rescale=1./255)
        test_datagen = ImageDataGenerator(rescale=1./255)
        #target dir, images resized to 150*150, use binary cross entropy
        train_generator = train_datagen.flow_from_directory(train_dir,target_size=(150, 150),batch_size=30,class_mode='binary')
        validation_generator = test_datagen.flow_from_directory(validation_dir,target_size=(150, 150),batch_size=30,class_mode='binary')
        test_generator = test_datagen.flow_from_directory(test_dir,target_size=(150,150),batch_size=30,class_mode=None)
        print("pre_processing")
        return train_generator,test_generator,validation_generator

    def train(model,train_generator,validation_generator,epochs=30, batch_size=64, model_name = 'cat-dog-classifier'):
        model.fit_generator(train_generator,steps_per_epoch=60,epochs=1,
                              validation_data=validation_generator,validation_steps=50)
        print("train")
        return model
        
    def evaluate(model,validation_generator,test_generator):
        STEP_SIZE_VALID=validation_generator.n//validation_generator.batch_size
        STEP_SIZE_TEST=test_generator.n//test_generator.batch_size
        model.evaluate_generator(validation_generator,steps=STEP_SIZE_VALID)
        test_generator.reset()
        predictions =model.predict_generator(test_generator,steps=STEP_SIZE_TEST,verbose=1)
        #val_preds =np.argmax(predictions,axis=1)
        #val_true = validation_generator.classes
        #cm = confusion_matrix(val_true, val_preds)
        #print(cm)
        #print(classification_report(val_true, val_preds))

        
        y_pred = []
        y_true = []
        #pred = model.predict(testX)
        for ty in predictions:
            #print("ty, y_pred: ", ty, ty.argmax(axis=0))
            y_pred.append(ty.argmax(axis=0))
        for ty in test_generator:
            #print("ty, y_true: ", ty, ty.argmax(axis=0))
            y_true.append(ty.argmax(axis=0))

        accuracy = accuracy_score(y_true, y_pred)
        print("Accuracy: {}%".format(100 * accuracy))

        precision = precision_score(y_true, y_pred, average="weighted")
        print("Precision: {}%".format(100 * precision))

        recall =recall_score(y_true, y_pred, average="weighted")
        print("Recall: {}%".format(100 * recall))

        f_score = f1_score(y_true, y_pred, average="weighted")
        print("f1_score: {}%".format(100 * f_score))
        '''
        print("Confusion Matrix:")
        confusion_matrix = metrics.confusion_matrix(y_true, y_pred)
        print(confusion_matrix)
        normalized_confusion_matrix = np.array(confusion_matrix, dtype=np.float32) / np.sum(confusion_matrix) * 100
        print("")
        print("Confusion matrix (normalised to % of total test data):")
        print(normalized_confusion_matrix)
        return y_pred, y_true, accuracy, precision, recall, f1_score, confusion_matrix, normalized_confusion_matrix

        '''
    def build_model(self):
     
        print(self.name)
        #self.model_file = model_file
        #print(self.model_file)
        if self.name == 'OrgNet':
            return self.OrgNet_model()
        if self.name == 'LeNet':
            return self.LeNet_model()
        elif self.name == 'MINST':
            return self.MINST_model()
        elif self.name == 'CIFAR10':
            return self.CIFAR10_model()
        elif self.name == 'AlexNet':
            return self.AlexNet_model()
        elif self.name == 'VGGNet':
            return self.VGGNet_model()
        elif self.name == 'GoogLeNet':
            return self.GoogLeNet_model()
        elif self.name == 'ResNet':
            return self.ResNet_model()
        elif self.name == 'ResNeXt':
            return self.build_ResNeXt()
        elif self.name == 'PlankNet':
            return self.build_PlankNet()
        elif self.name == 'CoapNet':
            print(self.name)
            return self.CoapNet_model()

    def AlexNet_model(img_shape = (150,150,3),num_classes=2):
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


    def LeNet_model(img_shape = (150,150,3),num_classes=2):
        #inputsize = self.input_width * self.input_height * self.input_channels
        LeNet = Sequential()
        LeNet.add(Conv2D(6,(5,5),activation='relu',input_shape=img_shape,
                         W_regularizer=l2(0.01)))
        LeNet.add(MaxPooling2D(pool_size=(2, 2)))
        LeNet.add(Conv2D(16,(5,5),activation='relu',input_shape=img_shape,
                          W_regularizer=l2(0.01)))
        LeNet.add(MaxPooling2D(pool_size=(2, 2)))
        LeNet.add(Flatten())
        LeNet.add(Dense(120))
        LeNet.add(Activation('tanh'))
        LeNet.add(Dropout(0.8))
        LeNet.add(Dense(84))
        LeNet.add(Activation('tanh'))
        LeNet.add(Dense(num_classes))
        LeNet.add(Activation('softmax'))
        LeNet.compile(loss='sparse_categorical_crossentropy',
                      optimizer=optimizers.adam(lr=0.01),metrics=['acc'])
        return LeNet

    def MNIST_model(img_shape = (150,150,3),num_classes=2):
        #inputsize = self.input_width * self.input_height * self.input_channels
        MNIST = Sequential()
        MNIST.add(Conv2D(32,(3,3),activation='relu',input_shape=img_shape, W_regularizer=l2(0.01)))
        MNIST.add(MaxPooling2D(pool_size=(2, 2)))
        MNIST.add(BatchNormalization())
        MNIST.add(Flatten())
        MNIST.add(Dense(128))
        MNIST.add(Activation('tanh'))
        MNIST.add(Dropout(0.8))
        MNIST.add(Dense(256))
        MNIST.add(Activation('tanh'))
        MNIST.add(Dense(num_classes))
        MNIST.add(Activation('softmax'))
        MNIST.compile(loss='sparse_categorical_crossentropy',
                      optimizer=optimizers.adam(lr=0.01),metrics=['acc'])
        return MNIST


    def OrgNet_model(img_shape =(150,150,3),num_classes=2):
        #inputsize = self.input_width * self.input_height * self.input_channels
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

    def CoapNet_model(img_shape = (150,150,3),num_classes=2):
        #inputsize = self.input_width * self.input_height * self.input_channels
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

    def CIFAR10_model(self,img_shape = (150,150,3),num_classes=10):
        #inputsize = self.input_width * self.input_height * self.input_channels
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

    def PlankNet_model(img_shape = (150,150,3),num_classes=2):
        #inputsize = self.input_width * self.input_height * self.input_channels
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
        PlankNet.add(Dense(4096, Activation ='tanh'))
        PlankNet.add(Dropout(0.5))
        PlankNet.add(Dense(4096, Activation ='tanh'))
        PlankNet.add(Dropout(0.5))
        PlankNet.Dense((num_classes))
        PlankNet.add(Activation('softmax'))
        PlankNet.compile(loss='sparse_categorical_crossentropy',
                      optimizer=optimizers.adam(lr=0.01),metrics=['acc'])
        return PlankNet

    
    def Googlenet_model(img_shape = (150,150,3),num_classes=2):
    # creates GoogLeNet a.k.a. Inception v1 (Szegedy, 2015)
        input = Input(shape=(3, 150, 150))

        input_pad = ZeroPadding2D(padding=(3, 3))(input)
        conv1_7x7_s2 = Conv2D(64, (7,7), strides=(2,2), padding='valid', activation='relu', name='conv1/7x7_s2', kernel_regularizer=l2(0.0002))(input_pad)
        conv1_zero_pad = ZeroPadding2D(padding=(1, 1))(conv1_7x7_s2)
        pool1_helper = PoolHelper()(conv1_zero_pad)
        pool1_3x3_s2 = MaxPooling2D(pool_size=(3,3), strides=(2,2), padding='valid', name='pool1/3x3_s2')(pool1_helper)
        pool1_norm1 = LRN(name='pool1/norm1')(pool1_3x3_s2)

        conv2_3x3_reduce = Conv2D(64, (1,1), padding='same', activation='relu', name='conv2/3x3_reduce', kernel_regularizer=l2(0.0002))(pool1_norm1)
        conv2_3x3 = Conv2D(192, (3,3), padding='same', activation='relu', name='conv2/3x3', kernel_regularizer=l2(0.0002))(conv2_3x3_reduce)
        conv2_norm2 = LRN(name='conv2/norm2')(conv2_3x3)
        conv2_zero_pad = ZeroPadding2D(padding=(1, 1))(conv2_norm2)
        pool2_helper = PoolHelper()(conv2_zero_pad)
        pool2_3x3_s2 = MaxPooling2D(pool_size=(3,3), strides=(2,2), padding='valid', name='pool2/3x3_s2')(pool2_helper)

        inception_3a_1x1 = Conv2D(64, (1,1), padding='same', activation='relu', name='inception_3a/1x1', kernel_regularizer=l2(0.0002))(pool2_3x3_s2)
        inception_3a_3x3_reduce = Conv2D(96, (1,1), padding='same', activation='relu', name='inception_3a/3x3_reduce', kernel_regularizer=l2(0.0002))(pool2_3x3_s2)
        inception_3a_3x3_pad = ZeroPadding2D(padding=(1, 1))(inception_3a_3x3_reduce)
        inception_3a_3x3 = Conv2D(128, (3,3), padding='valid', activation='relu', name='inception_3a/3x3', kernel_regularizer=l2(0.0002))(inception_3a_3x3_pad)
        inception_3a_5x5_reduce = Conv2D(16, (1,1), padding='same', activation='relu', name='inception_3a/5x5_reduce', kernel_regularizer=l2(0.0002))(pool2_3x3_s2)
        inception_3a_5x5_pad = ZeroPadding2D(padding=(2, 2))(inception_3a_5x5_reduce)
        inception_3a_5x5 = Conv2D(32, (5,5), padding='valid', activation='relu', name='inception_3a/5x5', kernel_regularizer=l2(0.0002))(inception_3a_5x5_pad)
        inception_3a_pool = MaxPooling2D(pool_size=(3,3), strides=(1,1), padding='same', name='inception_3a/pool')(pool2_3x3_s2)
        inception_3a_pool_proj = Conv2D(32, (1,1), padding='same', activation='relu', name='inception_3a/pool_proj', kernel_regularizer=l2(0.0002))(inception_3a_pool)
        inception_3a_output = Concatenate(axis=1, name='inception_3a/output')([inception_3a_1x1,inception_3a_3x3,inception_3a_5x5,inception_3a_pool_proj])

        inception_3b_1x1 = Conv2D(128, (1,1), padding='same', activation='relu', name='inception_3b/1x1', kernel_regularizer=l2(0.0002))(inception_3a_output)
        inception_3b_3x3_reduce = Conv2D(128, (1,1), padding='same', activation='relu', name='inception_3b/3x3_reduce', kernel_regularizer=l2(0.0002))(inception_3a_output)
        inception_3b_3x3_pad = ZeroPadding2D(padding=(1, 1))(inception_3b_3x3_reduce)
        inception_3b_3x3 = Conv2D(192, (3,3), padding='valid', activation='relu', name='inception_3b/3x3', kernel_regularizer=l2(0.0002))(inception_3b_3x3_pad)
        inception_3b_5x5_reduce = Conv2D(32, (1,1), padding='same', activation='relu', name='inception_3b/5x5_reduce', kernel_regularizer=l2(0.0002))(inception_3a_output)
        inception_3b_5x5_pad = ZeroPadding2D(padding=(2, 2))(inception_3b_5x5_reduce)
        inception_3b_5x5 = Conv2D(96, (5,5), padding='valid', activation='relu', name='inception_3b/5x5', kernel_regularizer=l2(0.0002))(inception_3b_5x5_pad)
        inception_3b_pool = MaxPooling2D(pool_size=(3,3), strides=(1,1), padding='same', name='inception_3b/pool')(inception_3a_output)
        inception_3b_pool_proj = Conv2D(64, (1,1), padding='same', activation='relu', name='inception_3b/pool_proj', kernel_regularizer=l2(0.0002))(inception_3b_pool)
        inception_3b_output = Concatenate(axis=1, name='inception_3b/output')([inception_3b_1x1,inception_3b_3x3,inception_3b_5x5,inception_3b_pool_proj])

        inception_3b_output_zero_pad = ZeroPadding2D(padding=(1, 1))(inception_3b_output)
        pool3_helper = PoolHelper()(inception_3b_output_zero_pad)
        pool3_3x3_s2 = MaxPooling2D(pool_size=(3,3), strides=(2,2), padding='valid', name='pool3/3x3_s2')(pool3_helper)

        inception_4a_1x1 = Conv2D(192, (1,1), padding='same', activation='relu', name='inception_4a/1x1', kernel_regularizer=l2(0.0002))(pool3_3x3_s2)
        inception_4a_3x3_reduce = Conv2D(96, (1,1), padding='same', activation='relu', name='inception_4a/3x3_reduce', kernel_regularizer=l2(0.0002))(pool3_3x3_s2)
        inception_4a_3x3_pad = ZeroPadding2D(padding=(1, 1))(inception_4a_3x3_reduce)
        inception_4a_3x3 = Conv2D(208, (3,3), padding='valid', activation='relu', name='inception_4a/3x3' ,kernel_regularizer=l2(0.0002))(inception_4a_3x3_pad)
        inception_4a_5x5_reduce = Conv2D(16, (1,1), padding='same', activation='relu', name='inception_4a/5x5_reduce', kernel_regularizer=l2(0.0002))(pool3_3x3_s2)
        inception_4a_5x5_pad = ZeroPadding2D(padding=(2, 2))(inception_4a_5x5_reduce)
        inception_4a_5x5 = Conv2D(48, (5,5), padding='valid', activation='relu', name='inception_4a/5x5', kernel_regularizer=l2(0.0002))(inception_4a_5x5_pad)
        inception_4a_pool = MaxPooling2D(pool_size=(3,3), strides=(1,1), padding='same', name='inception_4a/pool')(pool3_3x3_s2)
        inception_4a_pool_proj = Conv2D(64, (1,1), padding='same', activation='relu', name='inception_4a/pool_proj', kernel_regularizer=l2(0.0002))(inception_4a_pool)
        inception_4a_output = Concatenate(axis=1, name='inception_4a/output')([inception_4a_1x1,inception_4a_3x3,inception_4a_5x5,inception_4a_pool_proj])

        loss1_ave_pool = AveragePooling2D(pool_size=(5,5), strides=(3,3), name='loss1/ave_pool')(inception_4a_output)
        loss1_conv = Conv2D(128, (1,1), padding='same', activation='relu', name='loss1/conv', kernel_regularizer=l2(0.0002))(loss1_ave_pool)
        loss1_flat = Flatten()(loss1_conv)
        loss1_fc = Dense(1024, activation='relu', name='loss1/fc', kernel_regularizer=l2(0.0002))(loss1_flat)
        loss1_drop_fc = Dropout(rate=0.7)(loss1_fc)
        loss1_classifier = Dense(1000, name='loss1/classifier', kernel_regularizer=l2(0.0002))(loss1_drop_fc)
        loss1_classifier_act = Activation('softmax')(loss1_classifier)

        inception_4b_1x1 = Conv2D(160, (1,1), padding='same', activation='relu', name='inception_4b/1x1', kernel_regularizer=l2(0.0002))(inception_4a_output)
        inception_4b_3x3_reduce = Conv2D(112, (1,1), padding='same', activation='relu', name='inception_4b/3x3_reduce', kernel_regularizer=l2(0.0002))(inception_4a_output)
        inception_4b_3x3_pad = ZeroPadding2D(padding=(1, 1))(inception_4b_3x3_reduce)
        inception_4b_3x3 = Conv2D(224, (3,3), padding='valid', activation='relu', name='inception_4b/3x3', kernel_regularizer=l2(0.0002))(inception_4b_3x3_pad)
        inception_4b_5x5_reduce = Conv2D(24, (1,1), padding='same', activation='relu', name='inception_4b/5x5_reduce', kernel_regularizer=l2(0.0002))(inception_4a_output)
        inception_4b_5x5_pad = ZeroPadding2D(padding=(2, 2))(inception_4b_5x5_reduce)
        inception_4b_5x5 = Conv2D(64, (5,5), padding='valid', activation='relu', name='inception_4b/5x5', kernel_regularizer=l2(0.0002))(inception_4b_5x5_pad)
        inception_4b_pool = MaxPooling2D(pool_size=(3,3), strides=(1,1), padding='same', name='inception_4b/pool')(inception_4a_output)
        inception_4b_pool_proj = Conv2D(64, (1,1), padding='same', activation='relu', name='inception_4b/pool_proj', kernel_regularizer=l2(0.0002))(inception_4b_pool)
        inception_4b_output = Concatenate(axis=1, name='inception_4b/output')([inception_4b_1x1,inception_4b_3x3,inception_4b_5x5,inception_4b_pool_proj])

        inception_4c_1x1 = Conv2D(128, (1,1), padding='same', activation='relu', name='inception_4c/1x1', kernel_regularizer=l2(0.0002))(inception_4b_output)
        inception_4c_3x3_reduce = Conv2D(128, (1,1), padding='same', activation='relu', name='inception_4c/3x3_reduce', kernel_regularizer=l2(0.0002))(inception_4b_output)
        inception_4c_3x3_pad = ZeroPadding2D(padding=(1, 1))(inception_4c_3x3_reduce)
        inception_4c_3x3 = Conv2D(256, (3,3), padding='valid', activation='relu', name='inception_4c/3x3', kernel_regularizer=l2(0.0002))(inception_4c_3x3_pad)
        inception_4c_5x5_reduce = Conv2D(24, (1,1), padding='same', activation='relu', name='inception_4c/5x5_reduce', kernel_regularizer=l2(0.0002))(inception_4b_output)
        inception_4c_5x5_pad = ZeroPadding2D(padding=(2, 2))(inception_4c_5x5_reduce)
        inception_4c_5x5 = Conv2D(64, (5,5), padding='valid', activation='relu', name='inception_4c/5x5', kernel_regularizer=l2(0.0002))(inception_4c_5x5_pad)
        inception_4c_pool = MaxPooling2D(pool_size=(3,3), strides=(1,1), padding='same', name='inception_4c/pool')(inception_4b_output)
        inception_4c_pool_proj = Conv2D(64, (1,1), padding='same', activation='relu', name='inception_4c/pool_proj', kernel_regularizer=l2(0.0002))(inception_4c_pool)
        inception_4c_output = Concatenate(axis=1, name='inception_4c/output')([inception_4c_1x1,inception_4c_3x3,inception_4c_5x5,inception_4c_pool_proj])

        inception_4d_1x1 = Conv2D(112, (1,1), padding='same', activation='relu', name='inception_4d/1x1', kernel_regularizer=l2(0.0002))(inception_4c_output)
        inception_4d_3x3_reduce = Conv2D(144, (1,1), padding='same', activation='relu', name='inception_4d/3x3_reduce', kernel_regularizer=l2(0.0002))(inception_4c_output)
        inception_4d_3x3_pad = ZeroPadding2D(padding=(1, 1))(inception_4d_3x3_reduce)
        inception_4d_3x3 = Conv2D(288, (3,3), padding='valid', activation='relu', name='inception_4d/3x3', kernel_regularizer=l2(0.0002))(inception_4d_3x3_pad)
        inception_4d_5x5_reduce = Conv2D(32, (1,1), padding='same', activation='relu', name='inception_4d/5x5_reduce', kernel_regularizer=l2(0.0002))(inception_4c_output)
        inception_4d_5x5_pad = ZeroPadding2D(padding=(2, 2))(inception_4d_5x5_reduce)
        inception_4d_5x5 = Conv2D(64, (5,5), padding='valid', activation='relu', name='inception_4d/5x5', kernel_regularizer=l2(0.0002))(inception_4d_5x5_pad)
        inception_4d_pool = MaxPooling2D(pool_size=(3,3), strides=(1,1), padding='same', name='inception_4d/pool')(inception_4c_output)
        inception_4d_pool_proj = Conv2D(64, (1,1), padding='same', activation='relu', name='inception_4d/pool_proj', kernel_regularizer=l2(0.0002))(inception_4d_pool)
        inception_4d_output = Concatenate(axis=1, name='inception_4d/output')([inception_4d_1x1,inception_4d_3x3,inception_4d_5x5,inception_4d_pool_proj])

        loss2_ave_pool = AveragePooling2D(pool_size=(5,5), strides=(3,3), name='loss2/ave_pool')(inception_4d_output)
        loss2_conv = Conv2D(128, (1,1), padding='same', activation='relu', name='loss2/conv', kernel_regularizer=l2(0.0002))(loss2_ave_pool)
        loss2_flat = Flatten()(loss2_conv)
        loss2_fc = Dense(1024, activation='relu', name='loss2/fc', kernel_regularizer=l2(0.0002))(loss2_flat)
        loss2_drop_fc = Dropout(rate=0.7)(loss2_fc)
        loss2_classifier = Dense(1000, name='loss2/classifier', kernel_regularizer=l2(0.0002))(loss2_drop_fc)
        loss2_classifier_act = Activation('softmax')(loss2_classifier)

        inception_4e_1x1 = Conv2D(256, (1,1), padding='same', activation='relu', name='inception_4e/1x1', kernel_regularizer=l2(0.0002))(inception_4d_output)
        inception_4e_3x3_reduce = Conv2D(160, (1,1), padding='same', activation='relu', name='inception_4e/3x3_reduce', kernel_regularizer=l2(0.0002))(inception_4d_output)
        inception_4e_3x3_pad = ZeroPadding2D(padding=(1, 1))(inception_4e_3x3_reduce)
        inception_4e_3x3 = Conv2D(320, (3,3), padding='valid', activation='relu', name='inception_4e/3x3', kernel_regularizer=l2(0.0002))(inception_4e_3x3_pad)
        inception_4e_5x5_reduce = Conv2D(32, (1,1), padding='same', activation='relu', name='inception_4e/5x5_reduce', kernel_regularizer=l2(0.0002))(inception_4d_output)
        inception_4e_5x5_pad = ZeroPadding2D(padding=(2, 2))(inception_4e_5x5_reduce)
        inception_4e_5x5 = Conv2D(128, (5,5), padding='valid', activation='relu', name='inception_4e/5x5', kernel_regularizer=l2(0.0002))(inception_4e_5x5_pad)
        inception_4e_pool = MaxPooling2D(pool_size=(3,3), strides=(1,1), padding='same', name='inception_4e/pool')(inception_4d_output)
        inception_4e_pool_proj = Conv2D(128, (1,1), padding='same', activation='relu', name='inception_4e/pool_proj', kernel_regularizer=l2(0.0002))(inception_4e_pool)
        inception_4e_output = Concatenate(axis=1, name='inception_4e/output')([inception_4e_1x1,inception_4e_3x3,inception_4e_5x5,inception_4e_pool_proj])

        inception_4e_output_zero_pad = ZeroPadding2D(padding=(1, 1))(inception_4e_output)
        pool4_helper = PoolHelper()(inception_4e_output_zero_pad)
        pool4_3x3_s2 = MaxPooling2D(pool_size=(3,3), strides=(2,2), padding='valid', name='pool4/3x3_s2')(pool4_helper)

        inception_5a_1x1 = Conv2D(256, (1,1), padding='same', activation='relu', name='inception_5a/1x1', kernel_regularizer=l2(0.0002))(pool4_3x3_s2)
        inception_5a_3x3_reduce = Conv2D(160, (1,1), padding='same', activation='relu', name='inception_5a/3x3_reduce', kernel_regularizer=l2(0.0002))(pool4_3x3_s2)
        inception_5a_3x3_pad = ZeroPadding2D(padding=(1, 1))(inception_5a_3x3_reduce)
        inception_5a_3x3 = Conv2D(320, (3,3), padding='valid', activation='relu', name='inception_5a/3x3', kernel_regularizer=l2(0.0002))(inception_5a_3x3_pad)
        inception_5a_5x5_reduce = Conv2D(32, (1,1), padding='same', activation='relu', name='inception_5a/5x5_reduce', kernel_regularizer=l2(0.0002))(pool4_3x3_s2)
        inception_5a_5x5_pad = ZeroPadding2D(padding=(2, 2))(inception_5a_5x5_reduce)
        inception_5a_5x5 = Conv2D(128, (5,5), padding='valid', activation='relu', name='inception_5a/5x5', kernel_regularizer=l2(0.0002))(inception_5a_5x5_pad)
        inception_5a_pool = MaxPooling2D(pool_size=(3,3), strides=(1,1), padding='same', name='inception_5a/pool')(pool4_3x3_s2)
        inception_5a_pool_proj = Conv2D(128, (1,1), padding='same', activation='relu', name='inception_5a/pool_proj', kernel_regularizer=l2(0.0002))(inception_5a_pool)
        inception_5a_output = Concatenate(axis=1, name='inception_5a/output')([inception_5a_1x1,inception_5a_3x3,inception_5a_5x5,inception_5a_pool_proj])

        inception_5b_1x1 = Conv2D(384, (1,1), padding='same', activation='relu', name='inception_5b/1x1', kernel_regularizer=l2(0.0002))(inception_5a_output)
        inception_5b_3x3_reduce = Conv2D(192, (1,1), padding='same', activation='relu', name='inception_5b/3x3_reduce', kernel_regularizer=l2(0.0002))(inception_5a_output)
        inception_5b_3x3_pad = ZeroPadding2D(padding=(1, 1))(inception_5b_3x3_reduce)
        inception_5b_3x3 = Conv2D(384, (3,3), padding='valid', activation='relu', name='inception_5b/3x3', kernel_regularizer=l2(0.0002))(inception_5b_3x3_pad)
        inception_5b_5x5_reduce = Conv2D(48, (1,1), padding='same', activation='relu', name='inception_5b/5x5_reduce', kernel_regularizer=l2(0.0002))(inception_5a_output)
        inception_5b_5x5_pad = ZeroPadding2D(padding=(2, 2))(inception_5b_5x5_reduce)
        inception_5b_5x5 = Conv2D(128, (5,5), padding='valid', activation='relu', name='inception_5b/5x5', kernel_regularizer=l2(0.0002))(inception_5b_5x5_pad)
        inception_5b_pool = MaxPooling2D(pool_size=(3,3), strides=(1,1), padding='same', name='inception_5b/pool')(inception_5a_output)
        inception_5b_pool_proj = Conv2D(128, (1,1), padding='same', activation='relu', name='inception_5b/pool_proj', kernel_regularizer=l2(0.0002))(inception_5b_pool)
        inception_5b_output = Concatenate(axis=1, name='inception_5b/output')([inception_5b_1x1,inception_5b_3x3,inception_5b_5x5,inception_5b_pool_proj])

        pool5_7x7_s1 = AveragePooling2D(pool_size=(7,7), strides=(1,1), name='pool5/7x7_s2')(inception_5b_output)
        loss3_flat = Flatten()(pool5_7x7_s1)
        pool5_drop_7x7_s1 = Dropout(rate=0.4)(loss3_flat)
        loss3_classifier = Dense(num_classes, name='loss3/classifier', kernel_regularizer=l2(0.0002))(pool5_drop_7x7_s1)
        loss3_classifier_act = Activation('softmax', name='prob')(loss3_classifier)

        googlenet = Model(inputs=input, outputs=[loss1_classifier_act,loss2_classifier_act,loss3_classifier_act])
        sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
        googlenet.compile(optimizer=sgd, loss='categorical_crossentropy')

        return googlenet

    def ResNet_model(img_shape = (150,150,3),num_classes=2):
        #inputsize = self.input_width * self.input_height * self.input_channels

        base_model = ResNet50(weights= None, include_top=False, input_shape= img_shape)
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dropout(0.7)(x)
        predictions = Dense(num_classes, activation= 'softmax')(x)
        model = Model(inputs = base_model.input, outputs = predictions)

        return model
                
        

    

        

        
                  
        
        
        
        

        
    
