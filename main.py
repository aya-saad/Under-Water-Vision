#################################################################################################################
# A Modularized implementation for
# Image enhancement, segmentation, classification, evaluation and visualization
# Integrating all the algorithms stated above in one framework
# CONFIGURATION - LOAD_DATA - IMAGE_ENHANCEMENTS - IMAGE_SEGMENTATION -
# CLASS_BALANCING - FEATURE_IDENTIFICATION - CLASSIFICATION - EVALUATION_VISUALIZATION
# Author: Aya Saad
# Date created: 29 September 2019
# Project: AILARON
# funded by RCN FRINATEK IKTPLUSS program (project number 262701) and supported by NTNU AMOS
#
#################################################################################################################
import numpy as np
from configuration.utils import *
from configuration.config import get_config
from load_data.export_dataset import DatasetExporter
from load_data.load_dataset import DatasetLoader
from DNN.train_evaluate import *
from DNN.net import *

def build_hd5(data_dir, header_file, filename):
    #### CODE SNIPPET TO LOAD THE DATASET FROM THE DIRECTORY AND CREATE THE HD5 FILES ###########################
    dataset_loader = DatasetLoader(data_dir, header_file, filename)
    dataset_loader.get_classes_from_directory()
    dataset_loader.save_classes_to_file()
    dataset_loader.import_directory_structure()
    dataset_loader.save_data_to_file(dataset_loader.input_data, dataset_loader.filename)
    dataset_exporter = DatasetExporter(dataset_loader)
    dataset_exporter.export_train_test()
    dataset_exporter.export_CV()
    ## one file creation
    #file = os.path.join(data_dir,
    #                        "image_set_train.dat")  # the file that contains the list of images of the testing dataset along with their classes
    #dataset_exporter.build_hd5(file,input_width=3,input_height=3,input_channels=3,round='')
    ## building all the hd5 from a directory for the cross validation data
    dataset_exporter.build_all_hd5(data_dir)
    ##############################################################################################################
    return

def train_net(data_dir, model_path, log_file):
    name = 'AlexNet'
    n_splits = 1  # 10 for cross_validation, 1 for one time run
    myNet = AlexNet(name, input_width=244, input_height=244, input_channels=3,
                    num_classes=6, learning_rate=0.001,
                    momentum=0.09, keep_prob=0.5)
    fh = open(log_file, 'w')
    fh.write(name)
    print(name)
    round_num = ''
    out_test_hd5 = os.path.join(data_dir, 'image_set_test' + round_num + ".h5")  # + str(input_width) + '_db3'
    out_train_hd5 = os.path.join(data_dir, 'image_set_train' + round_num + ".h5") # str(input_width) + '_db3' +
    print(out_train_hd5)
    print(out_test_hd5)
    train_h5f = h5py.File(out_train_hd5, 'r')
    test_h5f = h5py.File(out_test_hd5, 'r')
    trainX = train_h5f['X']
    trainY = train_h5f['Y']
    print('trainX.shape ', trainX.shape, trainX[0])
    print('trainY.shape', trainY.shape, trainY[0])

    testX = test_h5f['X']
    testY = test_h5f['Y']
    print('testX.shape ', type(testX), testX.shape, testX[0])
    print('testY.shape', type(testY), testY.shape, testY[0])

    return

def main(config):
    np.random.seed(config.random_seed)
    prepare_dirs(config)
    header_file = config.data_dir + '/header.tfl.txt'
    log_file = os.path.join(config.model_dir, 'AlexNet.out')
    filename = config.data_dir + '/image_set.dat'

    print(config.model_dir, header_file, log_file, filename)

    # build_hd5(config.data_dir, header_file, filename)
    train_net(config.model_dir, header_file, log_file)


    return

if __name__ == '__main__':
    config, unparsed = get_config()
    main(config)