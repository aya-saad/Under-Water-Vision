"""TODO(pysil): Add a description here."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import cv2

import numpy as np
import tensorflow as tf
import tensorflow_datasets.public_api as tfds

IMG_SIZE = 64
IMG_SHAPE =(IMG_SIZE, IMG_SIZE, 3)
PYSIL_PATH = "/home/odask/tensorflow_datasets/pysil_org"
NUM_CLASSES = 8

_CITATION = """AILARON PROJECT PysilCam Images"""

_DESCRIPTION = """Images collected by AILARON TEAM with the pysilcam"""


class Pysil(tfds.core.GeneratorBasedBuilder):
  """PysilCam Dataset. """

  VERSION = tfds.core.Version('0.1.0')

  def _info(self):
    return tfds.core.DatasetInfo(
        builder=self,
        description=_DESCRIPTION,
        features=tfds.features.FeaturesDict({
            "image": tfds.features.Image(shape=IMG_SHAPE),
            "label": tfds.features.ClassLabel(num_classes=NUM_CLASSES),
        }),
        supervised_keys=("image", "label"),
        citation=_CITATION,
    )

  def _split_generators(self, dl_manager):
    """Returns SplitGenerators."""
    label_file = "batches.meta.txt"
    
    # Load the label names
    labels_path = "{}/{}".format(PYSIL_PATH, label_file)
    with tf.io.gfile.GFile(labels_path) as label_f:
        label_names = [name for name in label_f.read().split("\n") if name]
    self.info.features["label"].names = label_names
    
    # Define the splits
    def gen_filenames(splitdir):
        for label in os.listdir("{}/{}".format(PYSIL_PATH, splitdir)):
            if label != ".DS_Store":
                yield os.path.join(WHOI_PATH, splitdir, label)
    
    return [
        tfds.core.SplitGenerator(
            name=tfds.Split.TRAIN,
            num_shards=1,
            gen_kwargs={"filepaths": gen_filenames("train")}),
        tfds.core.SplitGenerator(
            name=tfds.Split.TEST,
            num_shards=1,
            gen_kwargs={"filepaths": gen_filenames("val")})
    ]

  def _generate_examples(self, filepaths):
      index = 0
      label_dict = dict(zip(self.info.features["label"].names, [i for i in range(len(self.info.features["label"].names))]))
      for path in filepaths:
          label_name = path.split("/")[-1]
          for img in os.listdir(path):
              img_path = "{}/{}".format(path, img)
              for labels, np_image in _load_data(label_name, label_dict, img_path):
                  record = dict(zip(["label"], labels))
                  record["image"] = np_image
                  yield record
                  #index += 1
    
    
    def _load_data(label_name, label_dict, path):
        """Yields (labels, np_image) tuples."""
        im = cv2.imread(path)
        #im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        img = (np.array(im)
                .reshape((3, IMG_SIZE, IMG_SIZE))
                .transpose(1, 2, 0))
        label = np.array([label_dict[label_name]])
        yield label, img

        """
        with tf.io.gfile.GFile(path, "rb") as f:
            data = f.read()
            img = (np.frombuffer(data, dtype=np.uint8)
                   .reshape((3, 128, 128))
                   .transpose((1, 2, 0)))
            label = np.array(label_dict[label_name])
            yield labels, img
        """

