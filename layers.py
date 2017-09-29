import tensorflow as tf
import tensorflow.contrib.slim as slim
import glog as log
from tensorflow.contrib.framework import add_arg_scope

#computes sparse softmax cross entropy between logits and labels
SE_loss = tf.nn.sparse_softmax_cross_entropy_with_logits

def int_shape(x):
  return list(map(int, x.get_shape()[1: ]))

def normalize(layer):
  return layer
  #return layer/127.5 - 1.

def denormalize(layer):
  return layer
  #return (layer + 1.)/2.

def _update_dict(layer_dict, scope, layer):
  name = "{}/{}".format(tf.get_variable_scope().name, scope)
  layer_dict[name] = layer

def image_from_paths(paths, shape, is_grayscale=True, seed=None):
  filename_queue = tf.train.string_input_producer(list(paths), shuffle=False, seed=seed)
  reader = tf.FixedLengthRecordReader(record_bytes=128*128*4)
  filename, data = reader.read(filename_queue)
  log.info(filename)
  log.info(data)
  image = tf.decode_raw(data, tf.float32)
  image = tf.reshape(image,[128,128,1])
  # print image

  #if is_grayscale:
  #  image = tf.image.rgb_to_grayscale(image)
  # print image
  #input("pause")
  image.set_shape(shape)
  return filename, tf.to_float(image)

@add_arg_scope
def resnet_block(
    inputs, scope, num_outputs=64, kernel_size=[3, 3],
    stride=[1, 1], padding="SAME", layer_dict={}):
  with tf.variable_scope(scope):
    layer = conv2d(
        inputs, num_outputs, kernel_size, stride,
        padding=padding, activation_fn=tf.nn.relu, scope="conv1")
    layer = conv2d(
        layer, num_outputs, kernel_size, stride,
        padding=padding, activation_fn=None, scope="conv2")
    outputs = tf.nn.relu(tf.add(inputs, layer))
  _update_dict(layer_dict, scope, outputs)
  return outputs

@add_arg_scope
def repeat(inputs, repetitions, layer, layer_dict={}, **kargv):
  outputs = slim.repeat(inputs, repetitions, layer, **kargv)
  _update_dict(layer_dict, kargv['scope'], outputs)
  return outputs

@add_arg_scope
def conv2d(inputs, num_outputs, kernel_size, stride,
           layer_dict={}, activation_fn=None,
           #weights_initializer=tf.random_normal_initializer(0, 0.001),
           weights_initializer=tf.contrib.layers.xavier_initializer(),
           scope=None, name="", **kargv):
  outputs = slim.conv2d(
      inputs, num_outputs, kernel_size,
      stride, activation_fn=activation_fn,
      weights_initializer=weights_initializer,
    biases_initializer=tf.constant_initializer(0.0), scope=scope, **kargv)
      #biases_initializer=tf.zeros_initializer(dtype=tf.float32), scope=scope, **kargv)
  if name:
    scope = "{}/{}".format(name, scope)
  _update_dict(layer_dict, scope, outputs)
  return outputs

@add_arg_scope
def max_pool2d(inputs, kernel_size=[3, 3], stride=[1, 1],
               layer_dict={}, scope=None, name="", **kargv):
  outputs = slim.max_pool2d(inputs, kernel_size, stride, **kargv)
  if name:
    scope = "{}/{}".format(name, scope)
  _update_dict(layer_dict, scope, outputs)
  return outputs

@add_arg_scope
def tanh(inputs, layer_dict={}, name=None, **kargv):
  outputs = tf.nn.tanh(inputs, name=name, **kargv)
  _update_dict(layer_dict, name, outputs)
  return outputs
