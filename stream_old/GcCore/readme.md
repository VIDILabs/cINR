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
Depending of you graphics card you may use differents compute capabilities. See [CUDA wikipedia](https://en.wikipedia.org/wiki/CUDA#GPUs_supported) to find out which ones are supported !

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
cd dependencies
mkdir build
cd build
cmake ..
make
```

Then go back to the **root** and build the project.

```
mkdir build
cd build
cmake ../src -DCMAKE_BUILD_TYPE=Release -DSHARED=ON -DCMAKE_CUDA_FLAGS="-arch=sm_80"
make all
```

##### CMAKE_BUILD_TYPE
* **Debug** add the option -g.
* **Release** add the option -O3.
##### SHARED
* if *ON* GcCore will be built as the shared library, if not set or *OFF*, it will be built as static library.
##### CMAKE_CUDA_FLAGS
* Define the compute capability (if not set, default value : arch=sm_30)
#### Did it work ?

A "root/bin" folder should have been created with the target build folder (debug / release) containing all the binaries built.

## Include in a project

### C++ project

To include GcCore in a C++ project you need to copy the folder "include/" to get all the header files and the folder "bin/" to get the binaries.

### Python project

To include GcCore in a Python project you need first to build the project as shared library then you have to copy the folder "bin/" where all the binaries are in your python project. Finally, you can load them in your python script.

## Documentation

You can find the doxygen documentation here : [Link](./doc/doxygen/html/index.html)