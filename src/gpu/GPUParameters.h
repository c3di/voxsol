#pragma once

// Solve Displacement Kernel parameters
#define CENTER_VERTEX_INDEX     13          // 1D index of the center vertex in the local subproblem
#define NEUMANN_OFFSET          9 * 27      // Offset to the start of the Neumann stress vector inside an equation block
#define UPDATES_PER_VERTEX      2           // Number of vertices that should be updated stochastically per worker 
#define EQUATION_ENTRY_SIZE     9 * 27 + 3  // 27 3x3 matrices and one 1x3 vector for Neumann stress
#define BLOCK_SIZE              6           // Number of threads in one block dimension (total threads per block is BLOCK_SIZE^3)
#define THREADS_PER_BLOCK       216         // Number of total threads per block

#define MAX_BLOCKS_PER_ITERATION 256        // Maximum number of regions to update per iteration of the solve displacement routine

#define NUM_LAUNCHES_BETWEEN_RESIDUAL_UPDATES 100
