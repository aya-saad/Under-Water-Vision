import tensorflow_datasets as tfds
#from tensorflow_datasets import whoi
import tensorflow as tf

# tfds works in both Eager and Graph modes
tf.compat.v1.enable_eager_execution()

# See available datasets
print(tfds.list_builders())

# Construct a tf.data.Dataset
ds_train = tfds.load(name="whoi", split="train", shuffle_files=True)

# Build your input pipeline
ds_train = ds_train.shuffle(1000).batch(128).prefetch(10)
for features in ds_train.take(1):
  image, label = features["image"], features["label"]
  print(features)