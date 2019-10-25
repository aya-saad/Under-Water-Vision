import os

import tensorflow as tf
from tensorflow import keras

model_name = '40000'
with tf.compat.v1.Session()(graph=tf.Graph()) as sess:
    tf.saved_model.loader.load(sess, ["serve"], model_name)

with tf.Graph().as_default():
    tf.random.set_seed(42)

with tf.compat.v1.Session() as sess:
    sess.run(tf.contrib.tpu.initialize_system())

new_model = tf.keras.models.load_model(filepath=model_name)
