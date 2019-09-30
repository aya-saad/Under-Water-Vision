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

def main(config):
    np.random.seed(config.random_seed)
    prepare_dirs(config)
    header_file = config.data_dir + '/header.tfl.txt'
    filename = config.data_dir + '/image_set.dat'
    print(config.data_dir, header_file, filename)
    dataset_loader = DatasetLoader(config.data_dir, header_file, filename)
    dataset_loader.get_classes_from_directory()
    dataset_loader.save_classes_to_file()
    dataset_loader.import_directory_structure()
    dataset_loader.save_data_to_file(dataset_loader.input_data, dataset_loader.filename)
    dataset_exporter = DatasetExporter(dataset_loader)
    dataset_exporter.export_train_test()
    dataset_exporter.export_CV()
    #file = os.path.join(config.data_dir,
    #                        "image_set_train.dat")  # the file that contains the list of images of the testing dataset along with their classes
    #dataset_exporter.build_hd5(file,input_width=3,input_height=3,input_channels=3,round='')
    dataset_exporter.build_all_hd5(config.data_dir)

    return

if __name__ == '__main__':
    config, unparsed = get_config()
    main(config)