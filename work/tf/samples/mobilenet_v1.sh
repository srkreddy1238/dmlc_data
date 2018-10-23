#!/bin/sh

for ii in mobilenet_v1_1.0_224 mobilenet_v1_1.0_192 mobilenet_v1_1.0_160 mobilenet_v1_1.0_128 mobilenet_v1_0.75_224 mobilenet_v1_0.75_192 mobilenet_v1_0.75_160 mobilenet_v1_0.75_128 mobilenet_v1_0.5_224 mobilenet_v1_0.5_192 mobilenet_v1_0.5_160 mobilenet_v1_0.5_128 mobilenet_v1_0.25_224 mobilenet_v1_0.25_192 mobilenet_v1_0.25_160 mobilenet_v1_0.25_128 ; do
    size=`echo $ii | rev | cut -d'_' -f1 | rev`
    python3 tf_model_test.py http://download.tensorflow.org/models/mobilenet_v1_2018_08_02/${ii}.tgz $size "input" "MobilenetV1/Predictions/Reshape_1"
done
