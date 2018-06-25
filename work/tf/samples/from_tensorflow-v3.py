"""
Compile TensorFlow Models
===================
**Author**: `Siva <https://github.com/srkreddy1238>`_

This article is an introductory tutorial to deploy Tensorflow models with NNVM.

For us to begin with, tensorflow module is required to be installed.

A quick solution is to install tensorlfow from

https://www.tensorflow.org/install/install_sources

"""
import nnvm
import tvm
import numpy as np

import os.path

# Tensorflow imports
import tensorflow as tf
from tensorflow.core.framework import graph_pb2
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import tensor_util


def _ProcessGraphDefParam(graph_def):
  """Type-checks and possibly canonicalizes `graph_def`."""
  if not isinstance(graph_def, graph_pb2.GraphDef):
    # `graph_def` could be a dynamically-created message, so try a duck-typed
    # approach
    try:
      old_graph_def = graph_def
      graph_def = graph_pb2.GraphDef()
      graph_def.MergeFrom(old_graph_def)
    except TypeError:
      raise TypeError('graph_def must be a GraphDef proto.')
  return graph_def

# Creates graph from saved graph_def.pb.
with tf.gfile.FastGFile(os.path.join(
        "./", 'inception_v3_2016_08_28_frozen-with_shapes.pb'), 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    graph = tf.import_graph_def(graph_def, name='')
    graph_def = _ProcessGraphDefParam(graph_def)


# we can load the graph as NNVM compatible model
sym, params = nnvm.frontend.from_tensorflow(graph_def)
print ("Import completed")
######################################################################
# Load a test image
# ---------------------------------------------

output_shape = (1, 299, 299, 3)


######################################################################
# Compile the model on NNVM
# ---------------------------------------------
# We should be familiar with the process right now.
import nnvm.compiler
target = 'llvm'

shape_dict = {'input': (1, 299, 299, 3)}
dtype_dict = {'input': 'float32'}
graph, lib, params = nnvm.compiler.build(sym, target, shape_dict, dtype=dtype_dict, params=params)

print ("Build completed")

lib.export_library("imagenet_tensorflow-v3.so")
with open("imagenet_tensorflow-v3.json", "w") as fo:
    fo.write(graph.json())
with open("imagenet_tensorflow-v3.params", "wb") as fo:
    fo.write(nnvm.compiler.save_param_dict(params))


exit(0)
