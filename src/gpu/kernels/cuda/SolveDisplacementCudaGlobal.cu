#include "stdafx.h"
#include <algorithm>
#include <random>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <device_functions.h>
#include <device_launch_parameters.h>
#include "gpu/CudaCommonFunctions.h"
#include "solution/Vertex.h"
#include "solution/samplers/BlockSampler.h"


#define MATRIX_ENTRY(rhsMatricesStartPointer, matrixIndex, row, col) rhsMatricesStartPointer[matrixIndex*9 + col*3 + row]

#define LHS_MATRIX_INDEX 13            // Position of the LHS matrix in the material config equations
#define EQUATION_ENTRY_SIZE 9 * 27 + 3 // 27 3x3 matrices and one 1x3 vector for Neumann stress
#define NEUMANN_OFFSET 9 * 27          // Offset to the start of the Neumann stress vector inside an equation block
#define UPDATES_PER_THREAD 100  // Number of vertices that should be updated stochastically per thread per kernel execution

#define DYN_ADJUSTMENT_MAX 0.01f

__device__
int getGlobalIdx_1D_3DGlobal() {
    return blockIdx.x * blockDim.x * blockDim.y * blockDim.z
        + threadIdx.z * blockDim.y * blockDim.x
        + threadIdx.y * blockDim.x + threadIdx.x;
}

__device__
int getGlobalIdx_3D_3DGlobal() {
    int blockId = blockIdx.x + blockIdx.y * gridDim.x
        + gridDim.x * gridDim.y * blockIdx.z;
    int threadId = blockId * (blockDim.x * blockDim.y * blockDim.z)
        + (threadIdx.z * (blockDim.x * blockDim.y))
        + (threadIdx.y * blockDim.x) + threadIdx.x;
    return threadId;
}

__device__ bool isInsideSolution(int offsetX, int offsetY, int offsetZ, const uint3 solutionDimensions) {
	return offsetX >= 0 && offsetX < solutionDimensions.x && offsetY >= 0 && offsetY < solutionDimensions.y && offsetZ >= 0 && offsetZ < solutionDimensions.z;
}

__device__ void buildRHSVectorForVertexGlobal(
    REAL* rhsVec, 
    Vertex* verticesOnGPU, 
    const REAL* matrices, 
    short centerCoordX, 
    short centerCoordY, 
    short centerCoordZ, 
    const uint3 solutionDimensions
) {
    int localNeighborIndex = 0;
	unsigned int globalNeighborIndex = 0;
	Vertex dummy;

    // Build RHS vector by multiplying each neighbor's displacement with its RHS matrix
    for (char localOffsetZ = 0; localOffsetZ <= 2; localOffsetZ++) {
        for (char localOffsetY = 0; localOffsetY <= 2; localOffsetY++) {
            for (char localOffsetX = 0; localOffsetX <= 2; localOffsetX++) {

                if (localOffsetZ == 1 && localOffsetY == 1 && localOffsetX == 1) {
                    //This is the center vertex that we're solving for, so skip it
                    continue;
                }

                unsigned short localNeighborCoordX = centerCoordX + localOffsetX - 1;
                unsigned short localNeighborCoordY = centerCoordY + localOffsetY - 1;
                unsigned short localNeighborCoordZ = centerCoordZ + localOffsetZ - 1;

                //Local problem size is always 3x3x3 vertices, regardless of solution size
                localNeighborIndex = localOffsetZ * 9 + localOffsetY * 3 + localOffsetX;
				globalNeighborIndex = solutionDimensions.y*solutionDimensions.x*localNeighborCoordZ + solutionDimensions.x*localNeighborCoordY + localNeighborCoordX;
				
				Vertex neighbor;
				if (isInsideSolution(localNeighborCoordX, localNeighborCoordY, localNeighborCoordZ, solutionDimensions)) {
					neighbor = verticesOnGPU[globalNeighborIndex];
				}
				else {
					neighbor = dummy;
				}

                // RHS[neighbor] * displacement[neighbor]
                rhsVec[0] += MATRIX_ENTRY(matrices, localNeighborIndex, 0, 0) * neighbor.x;
                rhsVec[0] += MATRIX_ENTRY(matrices, localNeighborIndex, 0, 1) * neighbor.y;
                rhsVec[0] += MATRIX_ENTRY(matrices, localNeighborIndex, 0, 2) * neighbor.z;

                rhsVec[1] += MATRIX_ENTRY(matrices, localNeighborIndex, 1, 0) * neighbor.x;
                rhsVec[1] += MATRIX_ENTRY(matrices, localNeighborIndex, 1, 1) * neighbor.y;
                rhsVec[1] += MATRIX_ENTRY(matrices, localNeighborIndex, 1, 2) * neighbor.z;

                rhsVec[2] += MATRIX_ENTRY(matrices, localNeighborIndex, 2, 0) * neighbor.x;
                rhsVec[2] += MATRIX_ENTRY(matrices, localNeighborIndex, 2, 1) * neighbor.y;
                rhsVec[2] += MATRIX_ENTRY(matrices, localNeighborIndex, 2, 2) * neighbor.z;
            }
        }
    }
}

__device__ const REAL* getPointerToMatricesForVertexGlobal(Vertex& vertex, const REAL* matConfigEquations) {
    unsigned int equationIndex = static_cast<unsigned int>(vertex.materialConfigId) * (EQUATION_ENTRY_SIZE);
    return &matConfigEquations[equationIndex];
}

__device__ void updateVertexGlobal(Vertex& vertexToUpdate, REAL* rhsVec, const REAL* matrices) {

    //Move to right side of equation and apply Neumann stress
    rhsVec[0] = -rhsVec[0] + matrices[NEUMANN_OFFSET];
    rhsVec[1] = -rhsVec[1] + matrices[NEUMANN_OFFSET + 1];
    rhsVec[2] = -rhsVec[2] + matrices[NEUMANN_OFFSET + 2];

    //rhsVec * LHS^-1
    REAL dx = MATRIX_ENTRY(matrices, LHS_MATRIX_INDEX, 0, 0) * rhsVec[0] +
        MATRIX_ENTRY(matrices, LHS_MATRIX_INDEX, 1, 0) * rhsVec[1] +
        MATRIX_ENTRY(matrices, LHS_MATRIX_INDEX, 2, 0) * rhsVec[2];
    REAL dy = MATRIX_ENTRY(matrices, LHS_MATRIX_INDEX, 0, 1) * rhsVec[0] +
        MATRIX_ENTRY(matrices, LHS_MATRIX_INDEX, 1, 1) * rhsVec[1] +
        MATRIX_ENTRY(matrices, LHS_MATRIX_INDEX, 2, 1) * rhsVec[2];
    REAL dz = MATRIX_ENTRY(matrices, LHS_MATRIX_INDEX, 0, 2) * rhsVec[0] +
        MATRIX_ENTRY(matrices, LHS_MATRIX_INDEX, 1, 2) * rhsVec[1] +
        MATRIX_ENTRY(matrices, LHS_MATRIX_INDEX, 2, 2) * rhsVec[2];

    //REAL diff = (abs(vertexToUpdate.x - dx) + abs(vertexToUpdate.y - dy) + abs(vertexToUpdate.z - dz)) * 0.3333333f;
    //if (diff > DYN_ADJUSTMENT_MAX) {
        // Perform dynamical adjustment, discarding any displacement deltas that are larger than the epsilon defined in DYN_ADJUSTMENT_MAX
        // this is to prevent occasional large errors caused by race conditions. Smaller errors are corrected over time by the stochastic updates
        // of surrounding vertices
    //    printf("Bad adjustment: %f diff for %f,%f,%f \n", diff, rhsVec[0], rhsVec[1], rhsVec[2]);
    //    return;
    //}

    vertexToUpdate.x = dx;
    vertexToUpdate.y = dy;
    vertexToUpdate.z = dz;
}

__device__ void updateVerticesStochasticallyGlobal(
    Vertex* verticesOnGPU, 
    const REAL* matConfigEquations, 
    curandState localRNGState, 
    const int3& blockOriginCoord, 
    const uint3 solutionDimensions
) {

    for (int i = 0; i < UPDATES_PER_THREAD; i++) {
        // There's a 1 vertex border around the problem area that shouldn't be updated, so choose something in the middle region
        int offsetX = blockOriginCoord.x + lroundf(curand_uniform(&localRNGState) * BLOCK_SIZE); 
		int offsetY = blockOriginCoord.y + lroundf(curand_uniform(&localRNGState) * BLOCK_SIZE);
		int offsetZ = blockOriginCoord.z + lroundf(curand_uniform(&localRNGState) * BLOCK_SIZE);

		if (!isInsideSolution(offsetX, offsetY, offsetZ, solutionDimensions)) {
			continue;
		}

		int offset = solutionDimensions.y*solutionDimensions.x*offsetZ + solutionDimensions.x*offsetY + offsetX;
		Vertex globalVertexToUpdate = verticesOnGPU[offset];

        if (globalVertexToUpdate.materialConfigId == static_cast<ConfigId>(0)) {
			// config id 0 should always be the case where the vertex is surrounded by empty cells, therefore not updateable
            continue;
        }

        REAL rhsVec[3] = { 0,0,0 };
        const REAL* matrices = getPointerToMatricesForVertexGlobal(globalVertexToUpdate, matConfigEquations);
        buildRHSVectorForVertexGlobal(rhsVec, verticesOnGPU, matrices, offsetX, offsetY, offsetZ, solutionDimensions);
        updateVertexGlobal(globalVertexToUpdate, rhsVec, matrices);
        verticesOnGPU[offset] = globalVertexToUpdate;
    }

}

__global__
void cuda_SolveDisplacementGlobal(
    Vertex* verticesOnGPU, 
    REAL* matConfigEquations, 
    const uint3 solutionDimensions,
    curandState* globalRNGStates, 
    const int3* blockOrigins
) {
    int3 blockOriginCoord = blockOrigins[blockIdx.x];
    curandState localRNGState = globalRNGStates[getGlobalIdx_1D_3DGlobal()];
	updateVerticesStochasticallyGlobal(verticesOnGPU, matConfigEquations, localRNGState, blockOriginCoord, solutionDimensions);
}

__global__
void cuda_init_curand_stateGlobal(curandState* rngState) {
    int id = getGlobalIdx_3D_3DGlobal();
    // seed, sequence number, offset, curandState
    curand_init(clock64(), id, 0, &rngState[id]);
}

__host__
curandState* initializeRNGStatesGlobal(int numConcurrentBlocks, dim3 threadsPerBlock) {
    int numThreads = numConcurrentBlocks * BLOCK_SIZE*BLOCK_SIZE*BLOCK_SIZE;
    curandState* rngStateOnGPU;
    cudaCheckSuccess(cudaMalloc(&rngStateOnGPU, sizeof(curandState) * numThreads));
    cuda_init_curand_stateGlobal <<< numConcurrentBlocks, threadsPerBlock >>> (rngStateOnGPU);
    cudaDeviceSynchronize();
    cudaCheckExecution();
    return rngStateOnGPU;
}

__host__
extern "C" void cudaLaunchSolveDisplacementKernelGlobal(
    Vertex* vertices, 
    REAL* matConfigEquations, 
    BlockSampler& sampler, 
    const uint3 solutionDims
) {
    cudaDeviceProp deviceProperties;
    cudaGetDeviceProperties(&deviceProperties, 0);

    // setup execution parameters
    dim3 threadsPerBlock(BLOCK_SIZE-2, BLOCK_SIZE-2, BLOCK_SIZE-2);
    int maxConcurrentBlocks = deviceProperties.multiProcessorCount*4; //TODO: Calculate this based on GPU max for # blocks

    // setup curand
    curandState* rngStateOnGPU = initializeRNGStatesGlobal(maxConcurrentBlocks, threadsPerBlock);

    int3* blockOrigins;
    cudaCheckSuccess(cudaMallocManaged(&blockOrigins, sizeof(int3) * maxConcurrentBlocks));
    
    for (int i = 0; i < 100; i++) {
        int numBlocks = sampler.generateNextBlockOrigins(blockOrigins, maxConcurrentBlocks);
        cuda_SolveDisplacementGlobal <<< numBlocks, threadsPerBlock >>>(vertices, matConfigEquations, solutionDims, rngStateOnGPU, blockOrigins);
        cudaDeviceSynchronize();
        cudaCheckExecution();
    }

    cudaCheckSuccess(cudaFree(blockOrigins));
    cudaCheckSuccess(cudaFree(rngStateOnGPU));
}


