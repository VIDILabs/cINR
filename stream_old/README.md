# GPU-Out-of-Core-Volume-Data
Out-of-core solution for large structured volume grids (voxels) access from GPU.

This is the code release attached to the paper [Interactive Visualization and On-Demand Processing of Large Volume Data: A Fully GPU-Based Out-of-Core Approach](https://hal.univ-reims.fr/hal-01705431) published in TVCG in 2020.

![shotCam](misc/teaser.png)

## Citation

If you use this code, please consider citing our work accordingly: 

```
@inproceedings{sarton:2020:goocvd,
 author = {Sarton, Jonathan and Courilleau, Nicolas and Remion, Yannick and Lucas, Laurent},
 journal = {IEEE Transactions on Visualization and Computer Graphics},
 title = {{Interactive Visualization and On-Demand Processing of Large Volume Data: A Fully GPU-Based Out-of-Core Approach}},
 year = {2020},
 volume = {26},
 number = {10},
 pages = {3008-3021},
 doi = {10.1109/TVCG.2019.2912752}
}
```

## Acknowledgments

This work was supported by the French national funds (PIA2’program “Intensive Computing and Numerical Simulation” call) under contract No. P112331-3422142 (**3DNeuroSecure** project). The purpose of this
project was to propose a collaborative solution to process and interactively visualize massive multi-scale data from ultra-high resolution 3D imaging. This secure solution also aims to break therapeutic innovation by allowing the exploitation of 3D images and complex data of large dimensions as part of applications framework linked to neurodegenerative diseases like Alzheimer’s. We would like to thank all the partners of the consortium led by Neoxia and in particular: the French Atomic Energy Commission (CEA), Tribvn, Archos and ESIA.

## Contact

Feel free to send an e-mail to `sarton[at]unistra.fr` or `laurent.lucas[at]univ-reims.fr` if you have any question.










# GcCore - Core C++/CUDA libraries for GPU Cache.

## Getting started
These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites

#### Windows
* **Visual studio 2015** (at least)
* **Cuda 8.0** (at least)

#### Linux

* **CMake 3.2** (at least)
* **Cuda 8.0** (at least)
* **GCC x.x** (depending on the Cuda version)

#### Mac OS not supported yet!

#### All
* A compute capabilty "compute_30,sm30" at least (NVidia Kepler).

## Building the project

###### Compute capabilities
Depending on your GPU you may use differents compute capabilities. See [CUDA wikipedia](https://en.wikipedia.org/wiki/CUDA#GPUs_supported) to find out which ones are supported !

### Windows

To build the project you need to go inside the folder **win_project** and double-clic on the "open_solution_smXX.bat" script
It will open the visual studio solution file with all the environment variables set.
The value *smXX* depend on your NVidia graphics card.

Then select the configuration (Release / Debug / Static or not) and clic on build.

To change compute capabilities, edit the "open_solution_smXX.bat" file and change the line
```
    SET CUDA_COMPUTE_CAPABILITY=compute_XX,sm_XX
```
by the expected compute capabilities.

### Linux

First of all download and compile the dependencies (LZ4)

```
cd GcCore/dependencies
mkdir build
cd build
cmake ..
make
```

Then go back to the **GcCore root** and build the project.

```
mkdir build
cd build
cmake ../src -DCMAKE_BUILD_TYPE=Release -DSHARED=ON -DCMAKE_CUDA_FLAGS="-arch=sm_80"
make all
```

##### CMAKE_BUILD_TYPE
* **Debug** add the debug options.
* **Release** add the performances optimisation options.
##### SHARED
* if *ON* GcCore will be built as the shared library, if not set or *OFF*, it will be built as static library.
##### CMAKE_CUDA_FLAGS
* Define the compute capability of your GPU (if not set, default value : arch=sm_30)
#### Did it work ?

A "GcCore/bin" folder should have been created with the target build folder (debug / release) containing all the binaries built.

## Include in a project

### C++ project

To include GcCore in a C++ project you need to copy the folder "include/" to get all the header files and the folder "bin/" to get the binaries.

### Python project

To include GcCore in a Python project you need first to build the project as shared library then you have to copy the folder "bin/" where all the binaries are in your python project. Finally, you can load them in your python script.

## Documentation

You can find the doxygen documentation here : [Link](./doc/doxygen/html/index.html)


# Exemple application : GPU DVR Ray-casting using the GcCore out-of-core cache system.

## Getting started

### Prerequisites
* **GLM library**

## Building the project

### Windows
...

### Linux
```
cd Applications/
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DSHARED=ON -DCMAKE_CUDA_FLAGS="-arch=sm_80"
make
```

##### CMAKE_BUILD_TYPE
* **Debug** add the debug options.
* **Release** add the performances optimisation options.
##### SHARED
* if *ON*, build as a shared library, if not set or *OFF*, it will be built as static library.
##### CMAKE_CUDA_FLAGS
* Define the compute capability of your GPU (if not set, default value : arch=sm_30)
#### Did it work ?

A "root/bin" folder should have been created with the target build folder (debug / release) containing all the binaries built.

## Data configuration
* Add a new volume data in the data/ folder with the following convention name :

    *data/name_of_the_volume_data/name_of_the_volume_data.extension*

    with possible reader *extension* : *raw*, *ima* (A new format and its associated reader can be added to GcCore libData FileFactory)
* Add a new config file in the config/ folder :

```
<Fields>
    <Field key="BrickSize" value="16"/>
    <Field key="BigBrickSizeX" value="1"/>
    <Field key="BigBrickSizeY" value="1"/>
    <Field key="BigBrickSizeZ" value="1"/>
    <Field key="VoxelCovering" value="1"/>
    <Field key="NumberEncodedBytes" value="1" />
    <Field key="NumberChannels" value="1" />
    <Field key="MultipleFiles" value="0" />
    <Field key="size_X" value="256" />
    <Field key="size_Y" value="256" />
    <Field key="size_Z" value="256" />
    <Field key="downScale_X" value="2" />
    <Field key="downScale_Y" value="2" />
    <Field key="downScale_Z" value="2" />
    <Field key="WorkingDirectory" value="../../data/" />
    <Field key="VolumeFile" value="name_of_the_volume_data.extension" />

    <!-- Display -->
    <Field key="ScreenWidth" value="1920" />
    <Field key="ScreenHeight" value="1080" />
    <Field key="TextureWidth" value="256" />
    <Field key="TextureHeight" value="256" />
</Fields>
```

    Settings can be changed (brick size ...)
    but the *VolumeFile* filed must be the same as the volume data with the extension.
