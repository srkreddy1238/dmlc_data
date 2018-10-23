"""
Compile Tensorflow Models
=========================
This article is an introductory tutorial to deploy tensorflow models with TVM.

For us to begin with, tensorflow python module is required to be installed.

Please refer to https://www.tensorflow.org/install
"""

# tvm and nnvm
import nnvm
import tvm

# os and numpy
import numpy as np
import os.path
import tarfile,sys

# Tensorflow imports
import tensorflow as tf
from tensorflow.core.framework import graph_pb2
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import tensor_util

# Tensorflow utility functions
import nnvm.testing.tf

target = 'llvm'
target_host = 'llvm'
layout = None
ctx = tvm.cpu(0)

def untar(fname):
    file_tar, file_tar_ext = os.path.splitext(fname)
    print(file_tar)
    if (fname.endswith("tgz")):
        tar = tarfile.open(fname)
        tar.extractall(path="./" + file_tar)
        tar.close()
        print("Extracted in Current Directory")
    else:
        print("Not a tar.gz file: '%s '" % sys.argv[0])


def get_workload(path):
    from mxnet.gluon.utils import download
    download(path, ".")

    tar_name = os.path.basename(sys.argv[1])
    untar(tar_name)

    file_tar, file_tar_ext = os.path.splitext(tar_name)
    model_name = file_tar + "/" + file_tar + "_frozen.pb"
    return model_name

def run_tvm_graph(graph_def, input_data, input_node, num_output=1, target='llvm'):
    """ Generic function to compile on nnvm and execute on tvm """

    layout = None
    if target == "cuda":
        layout = "NCHW"
    target_host = 'llvm'

    if isinstance(input_data, list):
        shape_dict = {}
        dtype_dict = {}
        for i, e in enumerate(input_node):
            shape_dict[e] = input_data[i].shape
            dtype_dict[e] = input_data[i].dtype
    else:
        shape_dict = {input_node: input_data.shape}
        dtype_dict = {input_node: input_data.dtype}

    sym, params = nnvm.frontend.from_tensorflow(graph_def, layout=layout, shape=shape_dict)
    graph, lib, params = nnvm.compiler.build(sym, target=target, target_host=target_host, shape=shape_dict,
                                             dtype=dtype_dict, params=params)

    ctx = tvm.context(target, 0)
    from tvm.contrib import graph_runtime
    m = graph_runtime.create(graph, lib, ctx)
    # set inputs
    if isinstance(input_data, list):
        for i, e in enumerate(input_node):
            m.set_input(e, tvm.nd.array(input_data[i].astype(input_data[i].dtype)))
    else:
        m.set_input(input_node, tvm.nd.array(input_data.astype(input_data.dtype)))

    m.set_input(**params)
    # execute
    m.run()
    # get outputs
    if num_output > 1:
        tvm_output_list = []
        for i in range(0, num_output):
            tvm_output = m.get_output(i)
            tvm_output_list.append(tvm_output.asnumpy())
        return tvm_output_list
    else:
        tvm_output = m.get_output(0)
        return tvm_output.asnumpy()

def run_tf_graph(sess, input_data, input_node, output_node):
    """ Generic function to execute tensorflow """

    tensor = sess.graph.get_tensor_by_name(output_node)

    if isinstance(input_data, list):
        input_dict = {}
        for i, e in enumerate(input_node):
            input_dict[e] = input_data[i]
    else:
        input_dict = {input_node: input_data}

    output_data = sess.run(tensor, input_dict)
    return output_data

with tf.Graph().as_default():
    model_name = get_workload(sys.argv[1])
    with tf.gfile.FastGFile(os.path.join("./", model_name), 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        graph = tf.import_graph_def(graph_def, name='')
        # Call the utility to import the graph definition into default graph.
        graph_def = nnvm.testing.tf.ProcessGraphDefParam(graph_def)

        in_node_name = sys.argv[3]
        out_node_name = sys.argv[4]

        in_shape = (1, int(sys.argv[2]), int(sys.argv[2]) , 3)
        shape_dict = {in_node_name: in_shape}
        data = np.random.uniform(size=in_shape).astype('float32')

        with tf.Session() as sess:
            # Add shapes to the graph.
            graph_def = nnvm.testing.tf.AddShapesToGraphDef(sess, out_node_name)
            tf_output = run_tf_graph(sess, data, in_node_name + ':0', out_node_name + ':0')
            tvm_output = run_tvm_graph(graph_def, data, in_node_name)
            np.testing.assert_allclose(np.squeeze(tvm_output), np.squeeze(tf_output), rtol=1e-5, atol=1e-5)
