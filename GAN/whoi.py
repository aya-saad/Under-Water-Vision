"""WHOI dataset."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import cv2

import numpy as np
import tensorflow as tf
import tensorflow_datasets.public_api as tfds

_CITATION = """\
@article{orenstein2015whoi,
  title={Whoi-plankton-a large scale fine grained visual recognition benchmark dataset for plankton classification},
  author={Orenstein, Eric C and Beijbom, Oscar and Peacock, Emily E and Sosik, Heidi M},
  journal={arXiv preprint arXiv:1510.00745},
  year={2015},
}"""

_DESCRIPTION = """WHOI dataset for Specialization Project of Oda Kiese, NTNU"""


class Whoi(tfds.core.GeneratorBasedBuilder):
  """WHOI dataset."""

  VERSION = tfds.core.Version('0.1.0')

  def _info(self):
    return tfds.core.DatasetInfo(
        builder=self,
        description=_DESCRIPTION,
        features=tfds.features.FeaturesDict({
            "image": tfds.features.Image(shape=(128, 128, 3)),
            "label": tfds.features.ClassLabel(num_classes=5),
        }),
        supervised_keys=("image", "label"),
        urls=['https://hdl.handle.net/10.1575/1912/7341'],
        citation=_CITATION,
    )


  def _split_generators(self, dl_manager):
    """Returns SplitGenerators."""
    label_file = "batches.meta.txt"
    whoi_path = "/home/odask/tensorflow_datasets/whoi_org"
    
    # Load the label names
    labels_path = "{}/{}".format(whoi_path, label_file)
    with tf.io.gfile.GFile(labels_path) as label_f:
        label_names = [name for name in label_f.read().split("\n") if name]
    self.info.features["label"].names = label_names
    
    # Define the splits
    def gen_filenames(splitdir):
        for label in os.listdir("{}/{}".format(whoi_path, splitdir)):
            if label != ".DS_Store":
                yield os.path.join(whoi_path, splitdir, label)
    ww
    return [
        tfds.core.SplitGenerator(
            name=tfds.Split.TRAIN,
            num_shards=9,
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
                # In order to run on the GPU, only return record.
                yield record
                #yield index, record
                #index += 1

def _load_data(label_name, label_dict, path):
    """Yields (labels, np_image) tuples."""
    # Default value is cv2.IMREAD_COLOR
    img = np.array(cv2.imread(path=path, flag=cv2.IMREAD_GRAYSCALE)
            .reshape((1, 128, 128))
            .transpose(1, 2, 0))
    label = np.array([label_dict[label_name]])
    yield label, img
    
    """
    The below is how the cifar10 is loading the images.
    with tf.io.gfile.GFile(path, "rb") as f:
        data = f.read()
        img = (np.frombuffer(data, dtype=np.uint8)
               .reshape((3, 128, 128))
               .transpose((1, 2, 0)))
        label = np.array(label_dict[label_name])
        yield labels, img
    """
