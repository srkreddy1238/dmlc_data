#!/usr/bin/python3

import os
import sys
import tarfile,sys

# tvm
import tvm
import tvm.relay as relay

# os and numpy
import numpy as np
import os.path

# Tensorflow imports
import tensorflow as tf
from tensorflow.core.framework import graph_pb2
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import tensor_util

# Tensorflow utility functions
import tvm.relay.testing.tf as tf_testing

target = 'llvm'
target_host = 'llvm'
layout = None
ctx = tvm.cpu(0)

models=[
            # Mobilenet V2
            {
                'name': 'mobilenet_v2_1.4_224',
                'dload_url':'https://storage.googleapis.com/mobilenet_v2/checkpoints/mobilenet_v2_1.4_224.tgz',
                'pb':'mobilenet_v2_1.4_224_frozen.pb',
                'in_node': 'input',
                'size': 224,
                'out_node': 'MobilenetV2/Predictions/Reshape_1'
            },
            {
                'name': 'mobilenet_v2_1.3_224',
                'dload_url':'https://storage.googleapis.com/mobilenet_v2/checkpoints/mobilenet_v2_1.3_224.tgz',
                'pb':'mobilenet_v2_1.3_224_frozen.pb',
                'in_node': 'input',
                'size': 224,
                'out_node': 'MobilenetV2/Predictions/Reshape_1'
            },
            {
                'name': 'mobilenet_v2_1.0_224',
                'dload_url':'https://storage.googleapis.com/mobilenet_v2/checkpoints/mobilenet_v2_1.0_224.tgz',
                'pb':'mobilenet_v2_1.0_224_frozen.pb',
                'in_node': 'input',
                'size': 224,
                'out_node': 'MobilenetV2/Predictions/Reshape_1'
            },
            {
                'name': 'mobilenet_v2_1.0_192',
                'dload_url':'https://storage.googleapis.com/mobilenet_v2/checkpoints/mobilenet_v2_1.0_192.tgz',
                'pb':'mobilenet_v2_1.0_192_frozen.pb',
                'in_node': 'input',
                'size': 192,
                'out_node': 'MobilenetV2/Predictions/Reshape_1'
            },
            {
                'name': 'mobilenet_v2_1.0_160',
                'dload_url':'https://storage.googleapis.com/mobilenet_v2/checkpoints/mobilenet_v2_1.0_160.tgz',
                'pb':'mobilenet_v2_1.0_160_frozen.pb',
                'in_node': 'input',
                'size': 160,
                'out_node': 'MobilenetV2/Predictions/Reshape_1'
            },
            {
                'name': 'mobilenet_v2_1.0_128',
                'dload_url':'https://storage.googleapis.com/mobilenet_v2/checkpoints/mobilenet_v2_1.0_128.tgz',
                'pb':'mobilenet_v2_1.0_128_frozen.pb',
                'in_node': 'input',
                'size': 128,
                'out_node': 'MobilenetV2/Predictions/Reshape_1'
            },
            {
                'name': 'mobilenet_v2_1.0_96',
                'dload_url':'https://storage.googleapis.com/mobilenet_v2/checkpoints/mobilenet_v2_1.0_96.tgz',
                'pb':'mobilenet_v2_1.0_96_frozen.pb',
                'in_node': 'input',
                'size': 96,
                'out_node': 'MobilenetV2/Predictions/Reshape_1'
            },
            {
                'name': 'mobilenet_v2_0.75_224',
                'dload_url':'https://storage.googleapis.com/mobilenet_v2/checkpoints/mobilenet_v2_0.75_224.tgz',
                'pb':'mobilenet_v2_0.75_224_frozen.pb',
                'in_node': 'input',
                'size': 224,
                'out_node': 'MobilenetV2/Predictions/Reshape_1'
            },
            {
                'name': 'mobilenet_v2_0.75_192',
                'dload_url':'https://storage.googleapis.com/mobilenet_v2/checkpoints/mobilenet_v2_0.75_192.tgz',
                'pb':'mobilenet_v2_0.75_192_frozen.pb',
                'in_node': 'input',
                'size': 192,
                'out_node': 'MobilenetV2/Predictions/Reshape_1'
            },
            {
                'name': 'mobilenet_v2_0.75_160',
                'dload_url':'https://storage.googleapis.com/mobilenet_v2/checkpoints/mobilenet_v2_0.75_160.tgz',
                'pb':'mobilenet_v2_0.75_160_frozen.pb',
                'in_node': 'input',
                'size': 160,
                'out_node': 'MobilenetV2/Predictions/Reshape_1'
            },
            {
                'name': 'mobilenet_v2_0.75_128',
                'dload_url':'https://storage.googleapis.com/mobilenet_v2/checkpoints/mobilenet_v2_0.75_128.tgz',
                'pb':'mobilenet_v2_0.75_128_frozen.pb',
                'in_node': 'input',
                'size': 128,
                'out_node': 'MobilenetV2/Predictions/Reshape_1'
            },
            {
                'name': 'mobilenet_v2_0.75_96',
                'dload_url':'https://storage.googleapis.com/mobilenet_v2/checkpoints/mobilenet_v2_0.75_96.tgz',
                'pb':'mobilenet_v2_0.75_96_frozen.pb',
                'in_node': 'input',
                'size': 96,
                'out_node': 'MobilenetV2/Predictions/Reshape_1'
            },
            {
                'name': 'mobilenet_v2_0.5_224',
                'dload_url':'https://storage.googleapis.com/mobilenet_v2/checkpoints/mobilenet_v2_0.5_224.tgz',
                'pb':'mobilenet_v2_0.5_224_frozen.pb',
                'in_node': 'input',
                'size': 224,
                'out_node': 'MobilenetV2/Predictions/Reshape_1'
            },
            {
                'name': 'mobilenet_v2_0.5_192',
                'dload_url':'https://storage.googleapis.com/mobilenet_v2/checkpoints/mobilenet_v2_0.5_192.tgz',
                'pb':'mobilenet_v2_0.5_192_frozen.pb',
                'in_node': 'input',
                'size': 192,
                'out_node': 'MobilenetV2/Predictions/Reshape_1'
            },
            {
                'name': 'mobilenet_v2_0.5_160',
                'dload_url':'https://storage.googleapis.com/mobilenet_v2/checkpoints/mobilenet_v2_0.5_160.tgz',
                'pb':'mobilenet_v2_0.5_160_frozen.pb',
                'in_node': 'input',
                'size': 160,
                'out_node': 'MobilenetV2/Predictions/Reshape_1'
            },
            {
                'name': 'mobilenet_v2_0.5_128',
                'dload_url':'https://storage.googleapis.com/mobilenet_v2/checkpoints/mobilenet_v2_0.5_128.tgz',
                'pb':'mobilenet_v2_0.5_128_frozen.pb',
                'in_node': 'input',
                'size': 128,
                'out_node': 'MobilenetV2/Predictions/Reshape_1'
            },
            {
                'name': 'mobilenet_v2_0.5_96',
                'dload_url':'https://storage.googleapis.com/mobilenet_v2/checkpoints/mobilenet_v2_0.5_96.tgz',
                'pb':'mobilenet_v2_0.5_96_frozen.pb',
                'in_node': 'input',
                'size': 96,
                'out_node': 'MobilenetV2/Predictions/Reshape_1'
            },
            {
                'name': 'mobilenet_v2_0.35_224',
                'dload_url':'https://storage.googleapis.com/mobilenet_v2/checkpoints/mobilenet_v2_0.35_224.tgz',
                'pb':'mobilenet_v2_0.35_224_frozen.pb',
                'in_node': 'input',
                'size': 224,
                'out_node': 'MobilenetV2/Predictions/Reshape_1'
            },
            {
                'name': 'mobilenet_v2_0.35_192',
                'dload_url':'https://storage.googleapis.com/mobilenet_v2/checkpoints/mobilenet_v2_0.35_192.tgz',
                'pb':'mobilenet_v2_0.35_192_frozen.pb',
                'in_node': 'input',
                'size': 192,
                'out_node': 'MobilenetV2/Predictions/Reshape_1'
            },
            {
                'name': 'mobilenet_v2_0.35_160',
                'dload_url':'https://storage.googleapis.com/mobilenet_v2/checkpoints/mobilenet_v2_0.35_160.tgz',
                'pb':'mobilenet_v2_0.35_160_frozen.pb',
                'in_node': 'input',
                'size': 160,
                'out_node': 'MobilenetV2/Predictions/Reshape_1'
            },
            {
                'name': 'mobilenet_v2_0.35_128',
                'dload_url':'https://storage.googleapis.com/mobilenet_v2/checkpoints/mobilenet_v2_0.35_128.tgz',
                'pb':'mobilenet_v2_0.35_128_frozen.pb',
                'in_node': 'input',
                'size': 128,
                'out_node': 'MobilenetV2/Predictions/Reshape_1'
            },
            {
                'name': 'mobilenet_v2_0.35_96',
                'dload_url':'https://storage.googleapis.com/mobilenet_v2/checkpoints/mobilenet_v2_0.35_96.tgz',
                'pb':'mobilenet_v2_0.35_96_frozen.pb',
                'in_node': 'input',
                'size': 96,
                'out_node': 'MobilenetV2/Predictions/Reshape_1'
            },
            # Mobilenet V1
            {
                'name': 'mobilenet_v1_0.25_128',
                'dload_url':'http://download.tensorflow.org/models/mobilenet_v1_2018_08_02/mobilenet_v1_0.25_128.tgz',
                'pb':'mobilenet_v1_0.25_128_frozen.pb',
                'in_node': 'input',
                'size': 128,
                'out_node': 'MobilenetV1/Predictions/Reshape_1'
            },
            {
                'name': 'mobilenet_v1_0.25_160',
                'dload_url':'http://download.tensorflow.org/models/mobilenet_v1_2018_08_02/mobilenet_v1_0.25_160.tgz',
                'pb':'mobilenet_v1_0.25_160_frozen.pb',
                'in_node': 'input',
                'size': 160,
                'out_node': 'MobilenetV1/Predictions/Reshape_1'
            },
            {
                'name': 'mobilenet_v1_0.25_192',
                'dload_url':'http://download.tensorflow.org/models/mobilenet_v1_2018_08_02/mobilenet_v1_0.25_192.tgz',
                'pb':'mobilenet_v1_0.25_192_frozen.pb',
                'in_node': 'input',
                'size': 192,
                'out_node': 'MobilenetV1/Predictions/Reshape_1'
            },
            {
                'name': 'mobilenet_v1_0.25_224',
                'dload_url':'http://download.tensorflow.org/models/mobilenet_v1_2018_08_02/mobilenet_v1_0.25_224.tgz',
                'pb':'mobilenet_v1_0.25_224_frozen.pb',
                'in_node': 'input',
                'size': 224,
                'out_node': 'MobilenetV1/Predictions/Reshape_1'
            },
            {
                'name': 'mobilenet_v1_0.5_128',
                'dload_url':'http://download.tensorflow.org/models/mobilenet_v1_2018_08_02/mobilenet_v1_0.5_128.tgz',
                'pb':'mobilenet_v1_0.5_128_frozen.pb',
                'in_node': 'input',
                'size': 128,
                'out_node': 'MobilenetV1/Predictions/Reshape_1'
            },
            {
                'name': 'mobilenet_v1_0.5_160',
                'dload_url':'http://download.tensorflow.org/models/mobilenet_v1_2018_08_02/mobilenet_v1_0.5_160.tgz',
                'pb':'mobilenet_v1_0.5_160_frozen.pb',
                'in_node': 'input',
                'size': 160,
                'out_node': 'MobilenetV1/Predictions/Reshape_1'
            },
            {
                'name': 'mobilenet_v1_0.5_192',
                'dload_url':'http://download.tensorflow.org/models/mobilenet_v1_2018_08_02/mobilenet_v1_0.5_192.tgz',
                'pb':'mobilenet_v1_0.5_192_frozen.pb',
                'in_node': 'input',
                'size': 192,
                'out_node': 'MobilenetV1/Predictions/Reshape_1'
            },
            {
                'name': 'mobilenet_v1_0.5_224',
                'dload_url':'http://download.tensorflow.org/models/mobilenet_v1_2018_08_02/mobilenet_v1_0.5_224.tgz',
                'pb':'mobilenet_v1_0.5_224_frozen.pb',
                'in_node': 'input',
                'size': 224,
                'out_node': 'MobilenetV1/Predictions/Reshape_1'
            },
            {
                'name': 'mobilenet_v1_0.75_128',
                'dload_url':'http://download.tensorflow.org/models/mobilenet_v1_2018_08_02/mobilenet_v1_0.75_128.tgz',
                'pb':'mobilenet_v1_0.75_128_frozen.pb',
                'in_node': 'input',
                'size': 128,
                'out_node': 'MobilenetV1/Predictions/Reshape_1'
            },
            {
                'name': 'mobilenet_v1_0.75_160',
                'dload_url':'http://download.tensorflow.org/models/mobilenet_v1_2018_08_02/mobilenet_v1_0.75_160.tgz',
                'pb':'mobilenet_v1_0.75_160_frozen.pb',
                'in_node': 'input',
                'size': 160,
                'out_node': 'MobilenetV1/Predictions/Reshape_1'
            },
            {
                'name': 'mobilenet_v1_0.75_192',
                'dload_url':'http://download.tensorflow.org/models/mobilenet_v1_2018_08_02/mobilenet_v1_0.75_192.tgz',
                'pb':'mobilenet_v1_0.75_192_frozen.pb',
                'in_node': 'input',
                'size': 192,
                'out_node': 'MobilenetV1/Predictions/Reshape_1'
            },
            {
                'name': 'mobilenet_v1_0.75_224',
                'dload_url':'http://download.tensorflow.org/models/mobilenet_v1_2018_08_02/mobilenet_v1_0.75_224.tgz',
                'pb':'mobilenet_v1_0.75_224_frozen.pb',
                'in_node': 'input',
                'size': 224,
                'out_node': 'MobilenetV1/Predictions/Reshape_1'
            },
            {
                'name': 'mobilenet_v1_1.0_128',
                'dload_url':'http://download.tensorflow.org/models/mobilenet_v1_2018_08_02/mobilenet_v1_1.0_128.tgz',
                'pb':'mobilenet_v1_1.0_128_frozen.pb',
                'in_node': 'input',
                'size': 128,
                'out_node': 'MobilenetV1/Predictions/Reshape_1'
            },
            {
                'name': 'mobilenet_v1_1.0_160',
                'dload_url':'http://download.tensorflow.org/models/mobilenet_v1_2018_08_02/mobilenet_v1_1.0_160.tgz',
                'pb':'mobilenet_v1_1.0_160_frozen.pb',
                'in_node': 'input',
                'size': 160,
                'out_node': 'MobilenetV1/Predictions/Reshape_1'
            },
            {
                'name': 'mobilenet_v1_1.0_192',
                'dload_url':'http://download.tensorflow.org/models/mobilenet_v1_2018_08_02/mobilenet_v1_1.0_192.tgz',
                'pb':'mobilenet_v1_1.0_192_frozen.pb',
                'in_node': 'input',
                'size': 192,
                'out_node': 'MobilenetV1/Predictions/Reshape_1'
            },
            {
                'name': 'mobilenet_v1_1.0_224',
                'dload_url':'http://download.tensorflow.org/models/mobilenet_v1_2018_08_02/mobilenet_v1_1.0_224.tgz',
                'pb':'mobilenet_v1_1.0_224_frozen.pb',
                'in_node': 'input',
                'size': 224,
                'out_node': 'MobilenetV1/Predictions/Reshape_1'
            },
            # Inception Resnet V2
            {
                'name': 'inception_resnet_v2',
                'dload_url':'http://download.tensorflow.org/models/inception_resnet_v2_2016_08_30.tar.gz',
                'ckpt_name': 'inception_resnet_v2_2016_08_30.ckpt',
                'pb':'frozen_inception_resnet_v2.pb',
                'pb_ops': '',
                'in_node':'input',
                'size':299,
                'out_node':'InceptionResnetV2/Logits/Predictions'
            },
            # Inception V4
            {
                'name': 'inception_v4',
                'dload_url':'http://download.tensorflow.org/models/inception_v4_2016_09_09.tar.gz',
                'ckpt_name': 'inception_v4.ckpt',
                'pb':'frozen_inception_v4.pb',
                'pb_ops': '',
                'in_node':'input',
                'size':299,
                'out_node':'InceptionV4/Logits/Predictions'
            },
            # Resnet V1
            {
                'name': 'resnet_v1_101',
                'dload_url':'http://download.tensorflow.org/models/resnet_v1_101_2016_08_28.tar.gz',
                'ckpt_name': 'resnet_v1_101.ckpt',
                'pb':'frozen_resnet_v1_101.pb',
                'pb_ops': ' --labels_offset=1 ',
                'in_node':'input',
                'size':224,
                'out_node':'resnet_v1_101/predictions/Reshape_1'
            },
            {
                'name': 'resnet_v1_152',
                'dload_url':'http://download.tensorflow.org/models/resnet_v1_152_2016_08_28.tar.gz',
                'ckpt_name': 'resnet_v1_152.ckpt',
                'pb':'frozen_resnet_v1_152.pb',
                'pb_ops': ' --labels_offset=1 ',
                'in_node':'input',
                'size':224,
                'out_node':'resnet_v1_152/predictions/Reshape_1'
            },
            {
                'name': 'resnet_v1_50',
                'dload_url':'http://download.tensorflow.org/models/resnet_v1_50_2016_08_28.tar.gz',
                'ckpt_name': 'resnet_v1_50.ckpt',
                'pb':'frozen_resnet_v1_50.pb',
                'pb_ops': ' --labels_offset=1 ',
                'in_node':'input',
                'size':224,
                'out_node':'resnet_v1_50/predictions/Reshape_1'
            },
            # Resnet V2
            {
                'name': 'resnet_v2_101',
                'dload_url':'http://download.tensorflow.org/models/resnet_v2_101_2017_04_14.tar.gz',
                'ckpt_name': 'resnet_v2_101.ckpt',
                'pb':'frozen_resnet_v2_101.pb',
                'pb_ops': '',
                'in_node':'input',
                'size':224,
                'out_node':'resnet_v2_101/predictions/Reshape_1'
            },
            {
                'name': 'resnet_v2_152',
                'dload_url':'http://download.tensorflow.org/models/resnet_v2_152_2017_04_14.tar.gz',
                'ckpt_name': 'resnet_v2_152.ckpt',
                'pb':'frozen_resnet_v2_152.pb',
                'pb_ops': '',
                'in_node':'input',
                'size':224,
                'out_node':'resnet_v2_152/predictions/Reshape_1'
            },
            {
                'name': 'resnet_v2_50',
                'dload_url':'http://download.tensorflow.org/models/resnet_v2_50_2017_04_14.tar.gz',
                'ckpt_name': 'resnet_v2_50.ckpt',
                'pb':'frozen_resnet_v2_50.pb',
                'pb_ops': '',
                'in_node':'input',
                'size':224,
                'out_node':'resnet_v2_50/predictions/Reshape_1'
            },
            # VGG
            {
                'name': 'vgg_16',
                'dload_url':'http://download.tensorflow.org/models/vgg_16_2016_08_28.tar.gz',
                'ckpt_name': 'vgg_16.ckpt',
                'pb':'frozen_vgg_16.pb',
                'pb_ops': ' --labels_offset=1 ',
                'in_node':'input',
                'size':224,
                'out_node':'vgg_16/fc8/squeezed'
            },
            {
                'name': 'vgg_19',
                'dload_url':'http://download.tensorflow.org/models/vgg_19_2016_08_28.tar.gz',
                'ckpt_name': 'vgg_19.ckpt',
                'pb':'frozen_vgg_19.pb',
                'pb_ops': ' --labels_offset=1 ',
                'in_node':'input',
                'size':224,
                'out_node':'vgg_19/fc8/squeezed'
            },
        ]


def setup_and_make_pb():
    if not os.path.isdir("./models"):
        os.system('git clone --recursive https://github.com/tensorflow/models')

    if not os.path.isdir("./data/models_pb"):
        os.mkdir("./data/models_pb")

    for model in models:
        if 'ckpt_name' in model:
            file_path = './data/models_pb/' + model['name'] + '_inf_graph.pb'
            if not os.path.exists(file_path):
                os.system('python3 ./models/research/slim/export_inference_graph.py \
                    --alsologtostderr \
                    --model_name=' + model['name'] + model['pb_ops'] + \
                    ' --output_file=' + file_path)

def untar(fname):
    file_tar, file_tar_ext = os.path.splitext(fname)
    if (fname.endswith("gz")):
        file_tar, file_tar_ext = os.path.splitext(file_tar)
        tar = tarfile.open('./data/models_dload/' +fname)
        tar.extractall(path="./data/models_dload/" + file_tar)
        tar.close()
    elif (fname.endswith("tgz")):
        tar = tarfile.open('./data/models_dload/' +fname)
        tar.extractall(path="./data/models_dload/" + file_tar)
        tar.close()
    else:
        print("Not a tar.gz file: '%s '" % fname)

def get_workload(path):
    if not os.path.isdir("./data/models_dload"):
        os.mkdir("./data/models_dload")

    from mxnet.gluon.utils import download
    download(path, "./data/models_dload/")

    tar_name = os.path.basename(path)
    file_tar, file_tar_ext = os.path.splitext(tar_name)
    if (tar_name.endswith("gz")):
        file_tar, file_tar_ext = os.path.splitext(file_tar)

    untar(tar_name)

    return './data/models_dload/' + file_tar

def download_models_from_repo():
    for model in models:
        tar_name = os.path.basename(model['dload_url'])
        file_tar, file_ext = os.path.splitext(tar_name)

        if 'ckpt_name' in model: # check point based models
            file_tar, file_ext = os.path.splitext(tar_name)
            if not os.path.exists('./data/models_dload/' + file_tar + '/' + model['ckpt_name']):
                folder = get_workload(model['dload_url'])
                model['folder'] = folder
            else:
                model['folder'] = './data/models_dload/' + file_tar
        else: # Frozen models already available
            if not os.path.exists('./data/models_dload/' + file_tar + '/' + model['pb']):
                folder = get_workload(model['dload_url'])
                model['folder'] = folder
            else:
                model['folder'] = './data/models_dload/' + file_tar

def gen_protobuf_from_ckpt():
    for model in models:
        if not os.path.exists(model['folder'] + '/' + model['pb']):
            print("Freeze:", model['name'])
            os.system('python3 -m tensorflow.python.tools.freeze_graph \
                --input_graph=./data/models_pb/' + model['name'] + '_inf_graph.pb' + \
                ' --input_checkpoint=' + model['folder'] + '/' + model['ckpt_name'] + \
                ' --input_binary=true --output_graph=' + model['folder'] + '/' + model['pb'] + \
                ' --output_node_names=' + model['out_node'])

def convert_to_list(x):
    if not isinstance(x, list):
        x = [x]
    return x

def run_tvm_graph(graph_def, input_data, input_node, num_output=1, target='llvm', out_names=None):
    """ Generic function to compile on relay and execute on tvm """
    input_data = convert_to_list(input_data)
    input_node = convert_to_list(input_node)

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

    sym, params = relay.frontend.from_tensorflow(graph_def,
                                                 layout=layout,
                                                 shape=shape_dict,
                                                 outputs=out_names)
    with relay.build_config(opt_level=3):
        graph, lib, params = relay.build(sym, target, params=params)

    ctx = tvm.context(target, 0)
    from tvm.contrib import graph_runtime
    m = graph_runtime.create(graph, lib, ctx)
    # set inputs
    for i, e in enumerate(input_node):
        m.set_input(e, tvm.nd.array(input_data[i].astype(input_data[i].dtype)))

    m.set_input(**params)
    # execute
    m.run()
    # get outputs
    assert out_names is None or num_output == len(out_names),"out_names: {} num_output: {}".format(
                                                              out_names, num_output)
    tvm_output_list = []
    for i in range(0, num_output):
        tvm_output = m.get_output(i)
        tvm_output_list.append(tvm_output.asnumpy())
    return tvm_output_list

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

def compile_test_tvm_tf():
    for model in models:
        with tf.Graph().as_default():
            print("Model:", model)
            model_name = model['pb']
            with tf.gfile.FastGFile(model['folder'] + '/' + model_name, 'rb') as f:
                graph_def = tf.GraphDef()
                graph_def.ParseFromString(f.read())
                graph = tf.import_graph_def(graph_def, name='')
                # Call the utility to import the graph definition into default graph.
                graph_def = tf_testing.ProcessGraphDefParam(graph_def)
    
                in_node_name = model['in_node']
                out_node_name = model['out_node']
    
                in_shape = (1, int(model['size']), int(model['size']) , 3)
                shape_dict = {in_node_name: in_shape}
                data = np.random.uniform(size=in_shape).astype('float32')

                with tf.Session() as sess:
                    # Add shapes to the graph.
                    graph_def = tf_testing.AddShapesToGraphDef(sess, out_node_name)
                    tf_output = run_tf_graph(sess, data, in_node_name + ':0', out_node_name + ':0')
                    tvm_output = run_tvm_graph(graph_def, data, in_node_name)
                    np.testing.assert_allclose(np.squeeze(tvm_output), np.squeeze(tf_output), rtol=1e-5, atol=1e-5)

# --------- MAIN ---------------------
setup_and_make_pb()
download_models_from_repo()
gen_protobuf_from_ckpt()
compile_test_tvm_tf()
