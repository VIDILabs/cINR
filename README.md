# cINR: A Cache-Accelerated INR framework for Interactive Visualization of Tera-Scale Data
*Machine learning has enabled the use of implicit neural representations (INRs) to efficiently compress and reconstruct massive scientific datasets. However, despite advances in fast INR rendering algorithms, INR-based rendering remains computationally expensive, as computing data values from an INR is significantly slower than reading them from GPU memory. This bottleneck currently restricts interactive INR visualization to professional workstations. To address this challenge, we introduce an INR rendering framework accelerated by a scalable, multi-resolution GPU cache capable of efficiently representing tera-scale datasets. By minimizing redundant data queries and prioritizing novel volume regions, our method reduces the number of INR computations per frame, achieving an average 5x speedup over the state-of-the-art INR rendering method while still maintaining high visualization quality. Coupled with existing hardware-accelerated INR compressors, our framework enables scientists to generate and compress massive datasets in situ on high-performance computing platforms and then interactively explore them on consumer-grade hardware post hoc.*

[Paper](https://arxiv.org/abs/2504.18001v3)

## Overview

[] TODO: Add training instructions for existing apps. \
[] TODO: Add docker instructions for included file. \
[] TODO: Create/update teaser image for the project. 

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

```bash
# git clone git@github.com:wilsonCernWq/open-volume-renderer.git
# cd open-volume-renderer/projects
git clone git@github.com:VIDILabs/cINR.git
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

All data requires a `scene.json` file for the OVR app, a `config.cfg` file for the Cache Manager and a `params.json` file for model params and the macrocell.

**Example scene, config, and params files are included in `cINR/data`.**  

The `config.cfg` file should be updated with the correct volume dimensions and path to the model weights.

The scene file should mirror the structure of our example scenes. 

```json
{ 
  // ... //
  "cache": {
    "config" : "<...>/config_miranda.cfg",
    // Cache dimensions, configure to fit into available VRAM
    "capacity": {
      "x": 30,
      "y": 30,
      "z": 30
    },
  },
  "macrocell": {
    "fileName": "<...>/miranda_params.json",
    //...
    //...
    }
  },
  // ... //
}
```
## Training [TODO]
*An instructional guide to using the provided training applications is coming soon*

## Running

If using VSCode, the following launch configuration may be added to `launch.json`.
```json
{
    "name": "(gdb) Evaluation App",
    "type": "cppdbg",
    "request": "launch",
    "program": "${workspaceFolder}/build/evalapp",
    "args": [
        "configs/scene_miranda.json", // Scene file
        "RM",                         // [Optional] Enable Ray Marching(RM)/Path Tracing(PT) data recording 
        "4000",                       // [Optional] Change the number of frames to record
        "1.0",                        // [Optional] Change the default target LoD scaling factor
        "83",                         // [Optional] Change the default phi for the directional light source
        "48",                        // [Optional] Change the default theta for the directional light source
        "1.5"                         // [Optional] Change the default light intensity 
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
../build/evalapp configs/scene_miranda.json
```

## Docker [TODO]

*Instructions coming soon*
