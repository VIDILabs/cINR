# cINR using (OVR)
{ project description/paper links }

## Overview

Cached Implicit Neural Radiance rendering demonstration.   

## Dependencies

- libaio-dev
    - on Ubuntu, install via `sudo apt-get install libaio-dev`
- CMake, CUDA 12+, and OptiX 7
- OptiX 7 SDK
    - download from http://developer.nvidia.com/optix and click "Get OptiX".
    - on linux, suggest to set the environment variable `OptiX_INSTALL_DIR` to wherever you installed the SDK.  
    `export OptiX_INSTALL_DIR=<wherever you installed OptiX 7 SDK>`
    - on windows, the installer should automatically put it into the right directory/

## Building

- Clone ~~`open-volume-renderer`~~ (this is no longer needed) and `cINR` and enter into the `stream` folder.

```bash
# git clone git@github.com:wilsonCernWq/open-volume-renderer.git
# cd open-volume-renderer/projects
git clone git@github.com:Deralik/cINR.git
cd cINR
git submodule update --init --recursive
```

- Build GcCore dependencies
```bash
cd stream/GcCore/dependencies
mkdir build
cd build
cmake ..
make
```

- Return to the root of this project
```bash
mkdir build
cd build
cmake .. \
    -DOVR_BUILD_OPENGL=TRUE \
    -DOVR_BUILD_DEVICE_OPTIX7=TRUE \
    -DOVR_BUILD_DEVICE_OSPRAY=FALSE \
    -DOVR_BUILD_MODULE_NNCACHE=TRUE \
    -DOptiX_INSTALL_DIR="<path-to-OptiX-SDK-folder>"
make
```

If your default CUDA compiler is older, you can specify the version of CUDA manually by adding the following flags to the cmake command.
```bash
-DCMAKE_CUDA_COMPILER=/usr/local/cuda-12/bin/nvcc -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-12 
```

## Data and Config

All data requires a `scene.json` file for the OVR, a `config.cfg` file for the Cache Manager and a `params.json` file for model params and the macrocell.

**Example scene, config, and params files are included in `cINR/data`. Relocate these to the data folder in OVR.**  

The `config.cfg` file should follow the same conventions as defined in `GcCore` with the `"WorkingDirectory"` containing the absolute path to `open-volume-renderer/data`.

The `VolumeFile` field us unused in this project. 

The scene file should mirror the structure of our example scenes. \
`cache.capacity` is also fully configerable based on the GPU.


```json
{ 
  // ... //
  "cache": {
    "config" : "<...>/config_miranda.cfg",
    // This cache at a brick size of 40 will be roughly 2.6GB
    "capacity": {
      "x": 13,
      "y": 13,
      "z": 13
    },
    "numLevels": 1 //Max desired LoD
  },
  "macrocell": {
    "fileName": "<...>/miranda_1024x1024x1024_float32/params.json",
    //...
    //...
    }
  },
  // ... //
}
```

## Running

If using VSCode, the following launch configuration may be added to `launch.json`.
```json
{
    "name": "(gdb) Evaluation App",
    "type": "cppdbg",
    "request": "launch",
    "program": "${workspaceFolder}/build/evalapp",
    "args": [
        "configs/scene_miranda.json",
        "nncache",
    ],
    "stopAtEntry": false,
    "cwd": "${workspaceFolder}/data",
    "environment": [
        {
            "name": "LD_LIBRARY_PATH",
            "value": "${env:LD_LIBRARY_PATH}:${workspaceFolder}/build"
        },
    ],
    "externalConsole": false,
    "MIMode": "gdb",
    "setupCommands": []
},
```

If running directly from the terminal, navigate to the root `data` folder.
```bash
../build/evalapp configs/scene_miranda.json nncache
```

## Docker

There is a Dockerfile provided for running the application in a container. This is useful for running on a machine without the necessary dependencies.

To build the image, navigate to the root of the project and run:
```bash
docker build -t wilsonovercloud/cacheinr:devel .
# docker tag cinr wilsonovercloud/cacheinr:devel
# docker push wilsonovercloud/cacheinr:devel
```

To run the container, use the following command:

```bash
xhost +si:localuser:root
docker run -ti --rm --runtime=nvidia --gpus all                         \
    -e CUDA_VISIBLE_DEVICES -e DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix \
    -v /mnt:/mnt -v /media:/media -v $(pwd):/workspace                  \
    wilsonovercloud/cacheinr:devel                                      \
    ./evalapp /workspace/data/configs/scene_miranda.json nncache RM -1
```
