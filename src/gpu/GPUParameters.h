#pragma once

// Solve Displacement Kernel parameters
#define CENTER_VERTEX_INDEX         13          // 1D index of the center vertex in the local subproblem
#define NEUMANN_OFFSET              (9 * 28)      // Offset to the start of the Neumann stress vector inside an equation block
#define EQUATION_ENTRY_SIZE         (9 * 28 + 3)  // 27 3x3 matrices and one 1x3 vector for Neumann stress
#define LHS_MATRIX_INDEX            27            // Index for the non-inverted LHS matrix, used to calculate residual error         

// Configurable parameters, experimentally determined optimum is given in the comment
#define UPDATES_PER_VERTEX          3           // 3    Number of vertices that should be updated stochastically per worker 
#define BLOCK_SIZE                  6           // 6    Number of threads in one block dimension (total threads per block is BLOCK_SIZE^3)
#define THREADS_PER_BLOCK           216         // 216  Number of total threads per block
#define MAX_CONCURRENT_BLOCKS       8           // 8    Max number of concurrent blocks per SM
#define MAX_BLOCKS_PER_ITERATION    2048         // 768  Maximum number of regions to update per iteration of the solve displacement routine

#define NUM_LAUNCHES_BETWEEN_RESIDUAL_UPDATES 500
