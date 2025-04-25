#!/bin/bash

FIELD=$1
if [ -z "$FIELD" ]; then
    echo "Usage: $0 <field>"
    echo "  Available fields:  H2, NH3, NO, N2O, O2, H, O, OH, HO2, H2O, H2O2, NO2, HNO, N, NNH, NH2, NH, H2NO, N2, temp, pressure, velx, vely, velz"
    exit 1
fi

IMAGE=wilsonovercloud/instantinr:devel

docker pull $IMAGE
docker run -u $(id -u):$(id -g) -ti --rm --runtime=nvidia --gpus all        \
    -e CUDA_VISIBLE_DEVICES -e DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix     \
    -v /mnt:/mnt -v /media:/media -v $(pwd):/workspace                      \
    $IMAGE vnr_cmd_train                                                    \
    --volume=/workspace/data/configs/scene_s3d_later_time/field_$FIELD.json \
    --network=/workspace/ovr/apps/example-model.json                        \
    --output=/workspace/params_$FIELD.json --max-num-steps=200000
