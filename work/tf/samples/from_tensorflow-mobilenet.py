"""
Compile Tensorflow Models
=========================
This article is an introductory tutorial to deploy tensorflow models with NNVM.

For us to begin with, tensorflow module is required to be installed.

A quick solution is to install tensorlfow from

https://www.tensorflow.org/install/install_sources
"""

import nnvm
import tvm
import numpy as np
import os.path
import re

# Tensorflow imports
import tensorflow as tf
from tensorflow.core.framework import graph_pb2
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import tensor_util

img_name = 'elephant-299.jpg'
model_name = 'model_with_output_shapes.pb'

######################################################################
# Some helper functions

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

######################################################################
# Creates graph from saved graph_def.pb.
with tf.gfile.FastGFile(os.path.join(
        "./", model_name), 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    graph = tf.import_graph_def(graph_def, name='')
    graph_def = _ProcessGraphDefParam(graph_def)


######################################################################
# Decode image
from PIL import Image
image = Image.open(img_name).resize((224, 224))

def transform_image(image):
    image = np.array(image).astype('float32')
    image = np.expand_dims(image, axis=0)
    return image

x = transform_image(image)
print('x', x.shape)
print('x type', x.dtype)

######################################################################
# Import the graph to NNVM
# -----------------

#print (graph_def)
sym, params = nnvm.frontend.from_tensorflow(graph_def)

######################################################################
# Now compile the graph through NNVM
import nnvm.compiler
target = 'llvm'
shape_dict = {'input': x.shape}
dtype_dict = {'input': x.dtype}
graph, lib, params = nnvm.compiler.build(sym, target, shape_dict, dtype=dtype_dict, params=params)

print ("Build Completed\n")


######################################################################
# Save the compilation output.
"""
lib.export_library("imagenet_tensorflow.so")
with open("imagenet_tensorflow.json", "w") as fo:
    fo.write(graph.json())
with open("imagenet_tensorflow.params", "wb") as fo:
    fo.write(nnvm.compiler.save_param_dict(params))
"""

######################################################################
# Execute the portable graph on TVM
# ---------------------------------
# Now, we would like to reproduce the same forward computation using TVM.
from tvm.contrib import graph_runtime
ctx = tvm.cpu(0)
dtype = 'float32'
m = graph_runtime.create(graph, lib, ctx)
# set inputs
m.set_input('input', tvm.nd.array(x.astype(dtype)))
m.set_input(**params)
# execute
m.run()
# get outputs
tvm_output = m.get_output(0, tvm.nd.empty(((1, 1001)), 'float32'))


######################################################################
# Process the output to human readable
# ------------------------------------
predictions = tvm_output.asnumpy()
predictions = np.squeeze(predictions)

print ("Predictions:", predictions)

top_k = predictions.argsort()[-5:][::-1]
print ("TOP TVM:", top_k)

"""
tf_output = np.load("tf_dump.txt.npy");

print ("TF: SHAPE:", tf_output.shape)
print ("TF:", tf_output)

top_k = tf_output.argsort()[-5:][::-1]
print ("TOP TF:", top_k)

np.testing.assert_allclose(tf_output, predictions, atol=1e-3, rtol=1e-3)
"""
exit(0)
