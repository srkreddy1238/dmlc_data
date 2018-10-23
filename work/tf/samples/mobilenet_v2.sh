#!/bin/sh

for ii in mobilenet_v2_1.4_224  mobilenet_v2_1.3_224 mobilenet_v2_1.0_224  mobilenet_v2_1.0_192 mobilenet_v2_1.0_160 mobilenet_v2_1.0_128 mobilenet_v2_1.0_96 mobilenet_v2_0.75_224 mobilenet_v2_0.75_192 mobilenet_v2_0.75_160 mobilenet_v2_0.75_128 mobilenet_v2_0.75_96 mobilenet_v2_0.5_224 mobilenet_v2_0.5_192 mobilenet_v2_0.5_160 mobilenet_v2_0.5_128 mobilenet_v2_0.5_96 mobilenet_v2_0.35_224 mobilenet_v2_0.35_192 mobilenet_v2_0.35_160 mobilenet_v2_0.35_128 mobilenet_v2_0.35_96; do
    size=`echo $ii | rev | cut -d'_' -f1 | rev`
    python3 tf_model_test.py https://storage.googleapis.com/mobilenet_v2/checkpoints/${ii}.tgz $size "input" "MobilenetV2/Predictions/Reshape_1"
done
