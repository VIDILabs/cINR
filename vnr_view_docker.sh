#!/bin/bash

xhost +si:localuser:root

PARAMS=$1
if [ -z "$PARAMS" ]; then
    echo "Usage: $0 <params.json>"
    exit 1
fi

IMAGE=wilsonovercloud/cacheinr:devel
docker pull $IMAGE
docker run -ti --rm --runtime=nvidia --gpus all                         \
    -e CUDA_VISIBLE_DEVICES -e DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix \
    -v /mnt:/mnt -v /media:/media -v $(pwd):/workspace                  \
    $IMAGE vnr_int_single --rendering-mode=0                            \
    --neural-volume=/workspace/$PARAMS                                  \
    --tfn=/workspace/data/configs/scene_s3d_later_time.json
