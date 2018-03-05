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
#include "gpu/sampling/ResidualVolume.h"


#define MATRIX_ENTRY(rhsMatricesStartPointer, matrixIndex, row, col) rhsMatricesStartPointer[matrixIndex*9 + col*3 + row]

#define LHS_MATRIX_INDEX 13            // Position of the LHS matrix in the material config equations
#define EQUATION_ENTRY_SIZE 9 * 27 + 3 // 27 3x3 matrices and one 1x3 vector for Neumann stress
#define NEUMANN_OFFSET 9 * 27          // Offset to the start of the Neumann stress vector inside an equation block
#define UPDATES_PER_THREAD 50  // Number of vertices that should be updated stochastically per thread per kernel execution

//#define DYN_ADJUSTMENT_MAX 0.01f

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

__device__ bool isInsideSolution(const uint3 coord, const uint3 solutionDimensions) {
    return coord.x < solutionDimensions.x && coord.y < solutionDimensions.y && coord.z < solutionDimensions.z;
}

__device__ void buildRHSVectorForVertexGlobal(
    REAL* rhsVec,
    Vertex* verticesOnGPU,
    const REAL* matrices,
    const uint3& centerCoord,
    const uint3& solutionDimensions
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
                uint3 localNeighborCoord;
                localNeighborCoord.x = centerCoord.x + localOffsetX - 1;
                localNeighborCoord.y = centerCoord.y + localOffsetY - 1;
                localNeighborCoord.z = centerCoord.z + localOffsetZ - 1;

                //Local problem size is always 3x3x3 vertices, regardless of solution size
                localNeighborIndex = localOffsetZ * 9 + localOffsetY * 3 + localOffsetX;
                globalNeighborIndex = solutionDimensions.y*solutionDimensions.x*localNeighborCoord.z + solutionDimensions.x*localNeighborCoord.y + localNeighborCoord.x;

                Vertex neighbor = dummy;

                if (isInsideSolution(localNeighborCoord, solutionDimensions)) {
                    neighbor = verticesOnGPU[globalNeighborIndex];
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

__device__ void updateVertexGlobalResidual(Vertex* vertexToUpdate, REAL* rhsVec, const REAL* matrices) {

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


#ifdef DYN_ADJUSTMENT_MAX
    REAL diff = (abs(vertexToUpdate->x - dx) + abs(vertexToUpdate->y - dy) + abs(vertexToUpdate->z - dz)) * 0.3333333f;
    if (diff > DYN_ADJUSTMENT_MAX) {
        // Perform dynamical adjustment, discarding any displacement deltas that are larger than the epsilon defined in DYN_ADJUSTMENT_MAX
        // this is to prevent occasional large errors caused by race conditions. Smaller errors are corrected over time by the stochastic updates
        // of surrounding vertices
        printf("Bad adjustment: %f diff for thread %i in block %i and bucket %i\n", diff, threadIdx.x, blockIdx.x, threadIdx.x / (blockDim.x / 2));
        return;
    }
#endif

    vertexToUpdate->x = dx;
    vertexToUpdate->y = dy;
    vertexToUpdate->z = dz;
}

__device__ void addResidualFromFullresVertex(
    const unsigned int x,
    const unsigned int y,
    const unsigned int z,
    REAL* residual,
    const REAL* matConfigEquations,
    const uint3& solutionDimensions,
    Vertex* verticesOnGPU
) {
    uint3 coord = { x, y, z };
    int fullresIndex = solutionDimensions.y*solutionDimensions.x*coord.z + solutionDimensions.x*coord.y + coord.x;
    if (!isInsideSolution(coord, solutionDimensions)) {
        // this vertex could lie outside the solution space because we expand the working block by 1 when gathering residuals, in this case residual is 0
        return;
    }
    Vertex globalFullresVertex = verticesOnGPU[fullresIndex];
    if (globalFullresVertex.materialConfigId == static_cast<ConfigId>(0)) {
        // config id 0 should always be the case where the vertex is surrounded by empty cells, therefore not updateable so residual is 0
        return;
    }
    REAL oldX = globalFullresVertex.x;
    REAL oldY = globalFullresVertex.y;
    REAL oldZ = globalFullresVertex.z;
    REAL rhsVec[3] = { 0,0,0 };
    const REAL* matrices = getPointerToMatricesForVertexGlobal(globalFullresVertex, matConfigEquations);
    buildRHSVectorForVertexGlobal(rhsVec, verticesOnGPU, matrices, coord, solutionDimensions);
    updateVertexGlobalResidual(&globalFullresVertex, rhsVec, matrices);

    // Get the magnitude of the displacement difference (residual)
    oldX = globalFullresVertex.x - oldX;
    oldY = globalFullresVertex.y - oldY;
    oldZ = globalFullresVertex.z - oldZ;
    oldX *= oldX;
    oldY *= oldY;
    oldZ *= oldZ;
    *residual += oldX + oldY + oldZ;
}

__device__ void updateResidualsLevelZeroGlobal(
    Vertex* verticesOnGPU,
    REAL* residualVolume,
    const REAL* matConfigEquations,
    const uint3& blockOriginCoord,
    const uint3& solutionDimensions,
    const LevelStats& levelZeroStats
) {
    // We want to find residuals for vertices bordering our BLOCK_SIZE area too, so -1, then project to level 0 with / 2
    uint3 vertexToUpdate;
    vertexToUpdate.x = (blockOriginCoord.x > 0 ? blockOriginCoord.x - 1 : 0) / 2;
    vertexToUpdate.y = (blockOriginCoord.y > 0 ? blockOriginCoord.y - 1 : 0) / 2;
    vertexToUpdate.z = (blockOriginCoord.z > 0 ? blockOriginCoord.z - 1 : 0) / 2;

    vertexToUpdate.z += threadIdx.x / (BLOCK_SIZE*BLOCK_SIZE);
    vertexToUpdate.y += (threadIdx.x - vertexToUpdate.z*BLOCK_SIZE*BLOCK_SIZE) / BLOCK_SIZE;
    vertexToUpdate.x += threadIdx.x % BLOCK_SIZE;

    if (vertexToUpdate.x >= levelZeroStats.sizeX ||
        vertexToUpdate.y >= levelZeroStats.sizeY ||
        vertexToUpdate.z >= levelZeroStats.sizeZ)
    {
        // Since level 0 has half the vertices some threads may be unnecessary
        return;
    }

    // Precompute the index of the residual we want to update on Level 0
    unsigned int residualIndex = vertexToUpdate.z * levelZeroStats.sizeX * levelZeroStats.sizeY + vertexToUpdate.y * levelZeroStats.sizeX + vertexToUpdate.x;

    // Project back down to fullres
    vertexToUpdate.x *= 2;
    vertexToUpdate.y *= 2;
    vertexToUpdate.z *= 2;

    REAL residual = asREAL(0.0);

    // Pool the residuals from the fullres level that contribute to this Level 0 vertex's residual
    addResidualFromFullresVertex(vertexToUpdate.x, vertexToUpdate.y, vertexToUpdate.z, &residual, matConfigEquations, solutionDimensions, verticesOnGPU);
    addResidualFromFullresVertex(vertexToUpdate.x + 1, vertexToUpdate.y, vertexToUpdate.z, &residual, matConfigEquations, solutionDimensions, verticesOnGPU);
    addResidualFromFullresVertex(vertexToUpdate.x, vertexToUpdate.y + 1, vertexToUpdate.z, &residual, matConfigEquations, solutionDimensions, verticesOnGPU);
    addResidualFromFullresVertex(vertexToUpdate.x + 1, vertexToUpdate.y + 1, vertexToUpdate.z, &residual, matConfigEquations, solutionDimensions, verticesOnGPU);

    addResidualFromFullresVertex(vertexToUpdate.x, vertexToUpdate.y, vertexToUpdate.z + 1, &residual, matConfigEquations, solutionDimensions, verticesOnGPU);
    addResidualFromFullresVertex(vertexToUpdate.x + 1, vertexToUpdate.y, vertexToUpdate.z + 1, &residual, matConfigEquations, solutionDimensions, verticesOnGPU);
    addResidualFromFullresVertex(vertexToUpdate.x, vertexToUpdate.y + 1, vertexToUpdate.z + 1, &residual, matConfigEquations, solutionDimensions, verticesOnGPU);
    addResidualFromFullresVertex(vertexToUpdate.x + 1, vertexToUpdate.y + 1, vertexToUpdate.z + 1, &residual, matConfigEquations, solutionDimensions, verticesOnGPU);

    residualVolume[residualIndex] = residual;
}

#define STAGGER
//#define SIMPLE
//#define BUCKET

__device__ void updateVertexStochasticallyGlobalResiduals(
    Vertex* verticesOnGPU,
    const REAL* matConfigEquations,
    curandState* localRNGState,
    const uint3& blockOriginCoord,
    const uint3& threadCoord,
    const uint3 solutionDimensions
) {
    if (!isInsideSolution(threadCoord, solutionDimensions)) {
        return;
    }
    Vertex globalVertexToUpdate;
    const int globalIndex = solutionDimensions.y*solutionDimensions.x*threadCoord.z + solutionDimensions.x*threadCoord.y + threadCoord.x;
    globalVertexToUpdate = verticesOnGPU[globalIndex];

    const REAL* matrices = getPointerToMatricesForVertexGlobal(globalVertexToUpdate, matConfigEquations);
    //const unsigned short bucketId = threadIdx.x / (blockDim.x / 2);
    //const unsigned short warpId = threadIdx.x / 32;
    
 
    for (int i = 0; i < UPDATES_PER_THREAD; i++) {
        REAL rhsVec[3] = { 0,0,0 };
        float diceRoll = curand_uniform(localRNGState);

#ifdef BUCKET
        // ping-pong between the two buckets of warps, effectively updating every second vertex per iteration
        // This strategy leads to divergence
        if (warpId % 2 == 1) {
            buildRHSVectorForVertexGlobal(rhsVec, verticesOnGPU, matrices, threadCoord, solutionDimensions);
            updateVertexGlobalResidual(&globalVertexToUpdate, rhsVec, matrices);
            verticesOnGPU[globalIndex] = globalVertexToUpdate;
        }
        
        if (warpId % 2 == 0) {
            buildRHSVectorForVertexGlobal(rhsVec, verticesOnGPU, matrices, threadCoord, solutionDimensions);
            updateVertexGlobalResidual(&globalVertexToUpdate, rhsVec, matrices);
            verticesOnGPU[globalIndex] = globalVertexToUpdate;
        }

        if (threadCoord.x == 4 && threadCoord.y == 1 && threadCoord.z == 2) {
            printf("Roll %f with RHS: %f %f %f with disp %f %f %f\n", diceRoll, rhsVec[0], rhsVec[1], rhsVec[2], globalVertexToUpdate.x, globalVertexToUpdate.y, globalVertexToUpdate.z);
        }
        
#endif
#ifdef STAGGER

        //This strategy is stable up to a dice threshold of 0.7
        if (diceRoll < 0.5) {
            buildRHSVectorForVertexGlobal(rhsVec, verticesOnGPU, matrices, threadCoord, solutionDimensions);
            updateVertexGlobalResidual(&globalVertexToUpdate, rhsVec, matrices);
            verticesOnGPU[globalIndex] = globalVertexToUpdate;
        }
        else {
            buildRHSVectorForVertexGlobal(rhsVec, verticesOnGPU, matrices, threadCoord, solutionDimensions);
            updateVertexGlobalResidual(&globalVertexToUpdate, rhsVec, matrices);
            verticesOnGPU[globalIndex] = globalVertexToUpdate;
        }
        
#endif
#ifdef SIMPLE

        // This strategy diverges
        buildRHSVectorForVertexGlobal(rhsVec, verticesOnGPU, matrices, threadCoord, solutionDimensions);
        updateVertexGlobalResidual(&globalVertexToUpdate, rhsVec, matrices);
        verticesOnGPU[globalIndex] = globalVertexToUpdate;

#endif
    }

}

__global__
void cuda_SolveDisplacementGlobalResiduals(
    Vertex* verticesOnGPU,
    REAL* matConfigEquations,
    REAL* residualVolume,
    const uint3 solutionDimensions,
    curandState* globalRNGStates,
    const uint3* blockOrigins,
    const LevelStats levelZeroStats
) {
    const uint3 blockOriginCoord = blockOrigins[blockIdx.x];
    if (blockOriginCoord.x >= solutionDimensions.x || blockOriginCoord.y >= solutionDimensions.y || blockOriginCoord.z >= solutionDimensions.z) {
        // Some blocks may have been set to an invalid value during the importance sampling phase if they overlap with some other block, these
        // should not be processed
        return;
    }
    curandState* localRNGState = &globalRNGStates[blockIdx.x * blockDim.x + threadIdx.x];

    const unsigned short HALF_BLOCK = BLOCK_SIZE / 2;
    const unsigned short bucket = threadIdx.x / (blockDim.x / 2); //either 0 or 1
    const unsigned short indexInBucket = threadIdx.x % (blockDim.x / 2);
    
    // Assign threads to coordinates according to a 3D checkerboard pattern so that all active warps are split into 2 groups
    // Group 1 will be executed first, updating every other vertex, then Group 2 will be executed afterward
    //  z == 0    z == 1   ...
    //  1 2 1 2   2 1 2 1
    //  2 1 2 1   1 2 1 2
    //  1 2 1 2   2 1 2 1
    //  2 1 2 1   1 2 1 2
    //  
    //
    uint3 coord;
    coord.z = indexInBucket / (HALF_BLOCK * BLOCK_SIZE);
    coord.y = (indexInBucket - coord.z*HALF_BLOCK * BLOCK_SIZE) / HALF_BLOCK;
    coord.x = (indexInBucket - HALF_BLOCK*(coord.y + BLOCK_SIZE*coord.z)) * 2;
    if (coord.z % 2 == bucket) {
        coord.x += coord.y % 2 != 0;
    }
    else {
        coord.x += coord.y % 2 == 0;
    }

    // Move the checkerboard pattern to the start of the block to be updated
    coord.x += blockOriginCoord.x;
    coord.y += blockOriginCoord.y;
    coord.z += blockOriginCoord.z;

    updateVertexStochasticallyGlobalResiduals(verticesOnGPU, matConfigEquations, localRNGState, blockOriginCoord, coord, solutionDimensions);

    __syncthreads();

    updateResidualsLevelZeroGlobal(verticesOnGPU, residualVolume, matConfigEquations, blockOriginCoord, solutionDimensions, levelZeroStats);
}

__global__
void cuda_init_curand_stateGlobal(curandState* rngState) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    // seed, sequence number, offset, curandState
    curand_init(42, id, 0, &rngState[id]);
}

__host__
extern "C" void cudaInitializeRNGStatesGlobal(curandState** rngStateOnGPU) {
    cudaDeviceProp deviceProperties;
    cudaGetDeviceProperties(&deviceProperties, 0);

    // setup execution parameters
    int threadsPerBlock = BLOCK_SIZE * BLOCK_SIZE * BLOCK_SIZE;
    int maxConcurrentBlocks = deviceProperties.multiProcessorCount * 4; //TODO: Calculate this based on GPU max for # blocks
    int numThreads = maxConcurrentBlocks *  threadsPerBlock;

    cudaCheckSuccess(cudaMalloc(rngStateOnGPU, sizeof(curandState) * numThreads));
    cuda_init_curand_stateGlobal << < maxConcurrentBlocks, threadsPerBlock >> > (*rngStateOnGPU);
    cudaDeviceSynchronize();
    cudaCheckExecution();
}

__host__
extern "C" void cudaLaunchSolveDisplacementKernelGlobalResiduals(
    Vertex* vertices,
    REAL* matConfigEquations,
    REAL* residualVolume,
    curandState* rngStateOnGPU,
    uint3* blockOrigins,
    const int numBlockOrigins,
    const uint3 solutionDims,
    const LevelStats* levelStats
) {
    cudaDeviceProp deviceProperties;
    cudaGetDeviceProperties(&deviceProperties, 0);

    // setup execution parameters
    int threadsPerBlock = BLOCK_SIZE * BLOCK_SIZE * BLOCK_SIZE;
    int maxConcurrentBlocks = deviceProperties.multiProcessorCount * 4; //TODO: Calculate this based on GPU max for # blocks
    int numIterations = std::max(numBlockOrigins / maxConcurrentBlocks, 1);
    
    // process all blocks in batches of maxConcurrentBlocks
    for (int i = 0; i < numIterations; i++) {
        uint3* currentBlockOrigins = &blockOrigins[i * maxConcurrentBlocks];
        int numBlocks = std::min(numBlockOrigins - i*maxConcurrentBlocks, maxConcurrentBlocks);
        cuda_SolveDisplacementGlobalResiduals << < numBlocks, threadsPerBlock >> >(vertices, matConfigEquations, residualVolume, solutionDims, rngStateOnGPU, currentBlockOrigins, levelStats[0]);
        cudaDeviceSynchronize();
        cudaCheckExecution();
    }
}


