
#from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import IPython.display as display
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os
import pathlib

AUTOTUNE = tf.data.experimental.AUTOTUNE

"""
def load_image(infilename):
    img = Image.open(infilename)
    img.load()
    npdata = np.asarray(img, dtype="int32")
    return npdata
"""

def rescale_image(filepath, filepath_out):
    height = 128
    width = 128
    img = Image.open(filepath)
    #img_resized = img.resize((height, width), Image.NEAREST)
    img_resized = img.resize((height, width), Image.BILINEAR) #gave a smoother result
    img_resized.save(filepath_out)


def rescale_corpus(dir, classes):
    for img_class in classes:
        img_class_path = "{}/{}".format(dir, img_class)
        img_class_path_out ="{}_scaled_bl/{}".format(corpus, img_class)

        if not os.path.exists(img_class_path_out):
            os.makedirs(img_class_path_out)

        for img in os.listdir(img_class_path):
            if img.split(".")[-1] == "png":
                filepath = "{}/{}".format(img_class_path, img)
                filepath_out = "{}/{}".format(img_class_path_out, img)
                rescale_image(filepath, filepath_out)
    return img_class_path_out


def load_corpus(data_dir):
    data_dir = pathlib.Path(data_dir)
    class_names = np.array([item.name for item in data_dir.glob('*') if item.name != ".DS_Store"])
    print(class_names)
    image_count = len(list(data_dir.glob('*/*.png')))
    print("Image count: {}".format(image_count))
    list_ds = tf.data.Dataset.list_files("{}/*/*".format(data_dir))
    return list_ds, class_names, image_count


def get_label(file_path):
    # convert the path to a list of path components
    parts = tf.strings.split(file_path, os.path.sep)
    # The second to last is the class-directory
    return parts[-2] == CLASS_NAMES


def decode_img(img):
    # convert the compressed string to a 3D uint8 tensor
    img = tf.image.decode_jpeg(img, channels=3)
    # Use `convert_image_dtype` to convert to floats in the [0,1] range.
    img = tf.image.convert_image_dtype(img, tf.float32)
    # resize the image to the desired size.
    return tf.image.resize(img, [IMG_WIDTH, IMG_HEIGHT])


def process_path(file_path):
    label = get_label(file_path)
    # load the raw data from the file as a string
    img = tf.io.read_file(file_path)
    img = decode_img(img)
    return img, label


def to_tfrecord(dataset):
    data_dir = pathlib.Path(dataset)
    IMAGE_COUNT = len(list(data_dir.glob('*/*.png')))


def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    # If the value is an eager tensor BytesList won't unpack a string from an EagerTensor.
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def serialize_example(feature0, feature1, feature2, feature3):
    feature = {
        'feature0': _int64_feature(feature0),
        'feature1': _float_feature(feature1),
        'feature2': _bytes_feature(feature2),
        'feature3': _bytes_feature(feature3),
    }

    # Create a Features message using tf.train.Example.
    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()


if __name__ == '__main__':
    #classes = ["Tintinnid", "Prorocentrum", "Dictyocha"]
    # For scaling corpus
    classes = ["zooplankton", "Tontonia_gracillima", "Strombidium_oculatum"]
    corpus = "2014"
    corpus_scaled = "2014_scaled_bl"

    #corpus_scaled = rescale_corpus(corpus, classes)

    IMG_WIDTH = 128
    IMG_HEIGHT = 128

    list_ds, CLASS_NAMES, IMAGE_COUNT = load_corpus(corpus_scaled)

    # Set `num_parallel_calls` so multiple images are loaded/processed in parallel.
    labeled_ds = list_ds.map(process_path, num_parallel_calls=AUTOTUNE)
    for image, label in labeled_ds.take(1):
        print("Image shape: ", image.numpy().shape)
        print("Label: ", label.numpy())

    for feature0, feature1, feature2, feature3 in dataset.take(1):
        serialized_example = serialize_example(feature0,
                                               feature1,
                                               feature2,
                                               tf.io.serialize_tensor(feature3))
        print(serialized_example)



#scale ned så høyde eller bredde er max 128
#lag tomt array med zeros og fyll inn bildet
#fyll opp de zerosene med random farger fra kantene på bildene/hjørnene på bildene

#lag et nice datasett uten å fylle opp







