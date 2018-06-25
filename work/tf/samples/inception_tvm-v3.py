"""
Run Compiled Tensorflow Model On TVM
===================
**Author**: `Siva <https://github.com/srkreddy1238>`_

This article is an introductory tutorial to deploy NNVM compiled Tensorflow models with TVM.

For us to begin with, tensorflow module is required to be installed.

A quick solution is to install tensorlfow from

https://www.tensorflow.org/install/install_sources

"""
from __future__ import absolute_import, print_function

import tvm
import numpy as np

# Global declarations of environment.

tgt_host="llvm"
# Change it to respective GPU if gpu is enabled Ex: cuda, opencl
tgt="llvm"

from tvm.contrib import graph_runtime
import argparse
import os.path
import re
import sys
import tarfile
import time
import tensorflow as tf
from tensorflow.python.framework import tensor_util

def load_labels(label_file):
    label = []
    proto_as_ascii_lines = tf.gfile.GFile(label_file).readlines()
    for l in proto_as_ascii_lines:
        label.append(l.rstrip())
    return label

def read_tensor_from_image_file(file_name, input_height=299, input_width=299,
				input_mean=0, input_std=255):
  input_name = "file_reader"
  output_name = "normalized"
  file_reader = tf.read_file(file_name, input_name)
    
  image_reader = tf.image.decode_jpeg(file_reader, channels = 3,
                                        name='jpeg_reader')
  float_caster = tf.cast(image_reader, tf.float32)
  dims_expander = tf.expand_dims(float_caster, 0)
  resized = tf.image.resize_bilinear(dims_expander, [input_height, input_width])
  normalized = tf.divide(tf.subtract(resized, [input_mean]), [input_std])
  tf.InteractiveSession()
  np_array = normalized.eval()
  tvm_array = tvm.nd.array(np_array)

  return tvm_array

loaded_lib = tvm.module.load("./imagenet_tensorflow-v3.so")
loaded_json = open("./imagenet_tensorflow-v3.json").read()
loaded_params = bytearray(open("./imagenet_tensorflow-v3.params", "rb").read())
module = graph_runtime.create(loaded_json, loaded_lib, tvm.cpu(0))
module.load_params(loaded_params)

#normalized = read_tensor_from_image_file("/media/sf_VMShare/cropped_panda.jpg")
normalized = read_tensor_from_image_file("/media/sf_VMShare/e1-299.jpg")
#normalized = read_tensor_from_image_file("/media/sf_VMShare/grace_hopper.jpg")

module.set_input('input', normalized)

#--DEBUG----------------------------
"""
out= module.debug_get_output("rsqrt0", out=tvm.nd.empty((32,), dtype='float32'))
#out= module.debug_get_output("broadcast_add0", out=tvm.nd.empty((32,), dtype='float32'))
tvm_out = out.asnumpy();
print("OUTPUT SHAPE:", tvm_out.shape)
print("OUTPUT:", tvm_out)

tf_ref_out = np.load("/home/srk/work/NN/TF/tensorflow/tf_dump.txt.npy")

print ("TF REF SHAPE:", tf_ref_out.shape)
print ("TF REF:", tf_ref_out)

if np.array_equal(tvm_out, tf_ref_out):
    print ("EQUAL")
else:
    print ("NOT EQUAL")

exit(0)
"""
#--DEBUG-END------------------------

out_shape = (1, 1001)

module.run()
out= module.get_output(0, out=tvm.nd.empty(out_shape))
# Print first 10 elements of output
print("OUTPUT:", out.asnumpy())

predictions = out.asnumpy()
results = np.squeeze(predictions)

top_k = results.argsort()[-5:][::-1]
labels = load_labels("/home/srk/work/NN/Inception-V3/imagenet_slim_labels.txt")
for i in top_k:
      print(labels[i], results[i])
