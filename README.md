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

## Examples

An example input is included consisting of a small aluminium material probe with silicon inclusions. A compressive load is applied to the top face of the cube. The bottom face is fixed in Z direction while two sides are fixed in X and Y directions respectively. 

It may be executed with output as a VTK file as follows:

```
stomech_solver -i "path/to/aluminium-silicon.xml" -o "path/to/output/folder/" --outputs "vtk"
```


