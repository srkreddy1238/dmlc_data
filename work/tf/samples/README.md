# Util to validate Resnet, VGG and InceptionV4

Tensorflow doesn't release Protobuf for all these models instead it's a check point.
Please follow https://github.com/tensorflow/models/tree/master/research/slim to
build it from check points.

Simple Steps:
=============

Step 1: 
-------
Download models tree
git clone git@github.com:tensorflow/models.git

Step 2: 
-------
Generate protobuf from slim repository

```bash
python export_inference_graph.py \
   --alsologtostderr \
   --model_name=<ModelName>_v3 \
   --output_file=<ModelName>_inf_graph.pb

Where ModelName can be 
    inception_resnet_v2
    inception_v4
    resnet_v1_101
    resnet_v1_152
    resnet_v1_50
    resnet_v2_101
    resnet_v2_152
    resnet_v2_50
    vgg_16
    vgg_19
```

**Note: Use additional option ```--labels_offset=1``` for resnet_v2 and vgg models.**

Step 3:
-------
Download checkpoints and extract them from models/research/slim

Step 4:
-------
Build freeze_graph utility from tensorflow source if required.

```bash
bazel build tensorflow/python/tools:freeze_graph

bazel-bin/tensorflow/python/tools/freeze_graph \
  --input_graph=<ModelName>_inf_graph.pb \
  --input_checkpoint=<CheckPointDir>/<ModelName>.ckpt \
  --input_binary=true --output_graph=frozen_<ModelName>.pb \
  --output_node_names=<OutputNodeName>

Where
    ModelName is same as above list.
    CheckPointDir is location where the ckpt files are available.
    OutputNodeName is the output nodename of the model.
```

OutputNodeName can be obtained from model summary by running below command on protobuf file before freeze.
```bash
    bazel-bin/tensorflow/tools/graph_transforms/summarize_graph \
        --in_graph=<ModelName>_inf_graph.pb 
```
