# Voxsol Mechanical Solver

A GPU-driven finite element solver used for the mechanical simulation of problems stemming from CT image stacks, FIB/SEM volumes or other 3D datasets. 

### Prerequisites

CUDA 10.1 and an Nvidia GPU with compute capability 6.0 or greater (GTX 9xx series or P100 series upwards).

The build system is currently tested on Windows only.

### Installing

First ensure CUDA 10.1 is installed.

After cloning the repository, run Cmake to generate the project files. Use the provided variables to set floating point precision and the compute capability of the target GPU.

## Tests

Tests are broken down into Unit tests and Integration tests. Both may be compiled as seperate executables.

Note that both test suites need to load data stored in the following directories:

```
test/data
test/src/unit/material/data
```

For this reason they cannot be run as standalone executables yet and should be executed from within the development environment. 

## Running 

The project creates a standalone executable that loads a problem from a specified .xml file:

```
stomech_solver -i "path/to/input.xml"
```

For help with command line options:

```
stomech_solver -h
```


