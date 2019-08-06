#include "stdafx.h"
#include <device_functions.h>
#include <device_launch_parameters.h>
#include "solution/Vertex.h"
#include "gpu/sampling/ResidualVolume.h"
#include "gpu/CudaCommonFunctions.h"
#include "gpu/GPUParameters.h"

#define MATRIX_ENTRY(rhsMatricesStartPointer, matrixIndex, row, col) __ldg(rhsMatricesStartPointer + matrixIndex*9 + col*3 + row) //matrixIndex*9 + col*3 + row

__constant__ uint3 c_solutionDimensions;
__constant__ uint3 c_residualDimensions;

__device__ bool isInsideSolutionSpace(const uint3 coord) {
    return coord.x < c_solutionDimensions.x && coord.y < c_solutionDimensions.y && coord.z < c_solutionDimensions.z;
}

__device__ bool isInsideResidualLeveLZero(const uint3 coord) {
    return coord.x < c_residualDimensions.x && coord.y < c_residualDimensions.y && coord.z < c_residualDimensions.z;
}

__device__ const REAL* getPointerToMatricesForVertexGlobalResid(volatile Vertex* vertex, const REAL* matConfigEquations) {
    unsigned int equationIndex = static_cast<unsigned int>(vertex->materialConfigId) * (EQUATION_ENTRY_SIZE);
    return &matConfigEquations[equationIndex];
}

__device__
int getGlobalIdx_3D_3D() {
    int blockId = blockIdx.x + blockIdx.y * gridDim.x
        + gridDim.x * gridDim.y * blockIdx.z;
    int threadId = blockId * (blockDim.x * blockDim.y * blockDim.z)
        + (threadIdx.z * (blockDim.x * blockDim.y))
        + (threadIdx.y * blockDim.x) + threadIdx.x;
    return threadId;
}

__device__ void buildRHSVectorForVertex(
    Vertex* verticesOnGPU,
    REAL* rhsVec,
    const REAL* __restrict__ matrices,
    const uint3& globalCenterCoord
) {   
    uint3 neighborCoord = {globalCenterCoord.x, globalCenterCoord.y, globalCenterCoord.z};
    Vertex dummy;

    for (char z = -1; z < 2; z++) {
        neighborCoord.z = globalCenterCoord.z + z;
        for (char y = -1; y < 2; y++) {
            neighborCoord.y = globalCenterCoord.y + y;
            for (char x = -1; x < 2; x++) {
                neighborCoord.x = globalCenterCoord.x + x;

                if (z == 0 && y == 0 && x == 0) {
                    continue;
                }

                const int localNeighborIndex = (z+1) * 9 + (y+1) * 3 + (x+1);  
                Vertex* __restrict__ neighbor = &dummy;

                if (isInsideSolutionSpace(neighborCoord)) {
                    const int globalIndex = c_solutionDimensions.y*c_solutionDimensions.x*neighborCoord.z + c_solutionDimensions.x*neighborCoord.y + neighborCoord.x;
                    neighbor = &verticesOnGPU[globalIndex];
                }

                const REAL nx = neighbor->x;
                const REAL ny = neighbor->y;
                const REAL nz = neighbor->z;

                rhsVec[0] += MATRIX_ENTRY(matrices, localNeighborIndex, 0, 0) * nx;
                rhsVec[0] += MATRIX_ENTRY(matrices, localNeighborIndex, 0, 1) * ny;
                rhsVec[0] += MATRIX_ENTRY(matrices, localNeighborIndex, 0, 2) * nz;

                rhsVec[1] += MATRIX_ENTRY(matrices, localNeighborIndex, 1, 0) * nx;
                rhsVec[1] += MATRIX_ENTRY(matrices, localNeighborIndex, 1, 1) * ny;
                rhsVec[1] += MATRIX_ENTRY(matrices, localNeighborIndex, 1, 2) * nz;

                rhsVec[2] += MATRIX_ENTRY(matrices, localNeighborIndex, 2, 0) * nx;
                rhsVec[2] += MATRIX_ENTRY(matrices, localNeighborIndex, 2, 1) * ny;
                rhsVec[2] += MATRIX_ENTRY(matrices, localNeighborIndex, 2, 2) * nz;
            }
        }
    }
    
}

__device__ void updateVertexResidual(
    REAL* residualsOnGPU,
    REAL* rhsVec,
    const REAL* __restrict__ matrices,
    const uint3& globalCenterCoord,
    Vertex* vertexToUpdate
) {
    const int residualIndex = globalCenterCoord.z * c_residualDimensions.y * c_residualDimensions.x + globalCenterCoord.y * c_residualDimensions.x + globalCenterCoord.x;

    rhsVec[0] = -rhsVec[0] + __ldg(matrices + NEUMANN_OFFSET);
    rhsVec[1] = -rhsVec[1] + __ldg(matrices + NEUMANN_OFFSET + 1);
    rhsVec[2] = -rhsVec[2] + __ldg(matrices + NEUMANN_OFFSET + 2);

    REAL dx = MATRIX_ENTRY(matrices, CENTER_VERTEX_INDEX, 0, 0) * rhsVec[0] +
        MATRIX_ENTRY(matrices, CENTER_VERTEX_INDEX, 1, 0) * rhsVec[1] +
        MATRIX_ENTRY(matrices, CENTER_VERTEX_INDEX, 2, 0) * rhsVec[2];
    REAL dy = MATRIX_ENTRY(matrices, CENTER_VERTEX_INDEX, 0, 1) * rhsVec[0] +
        MATRIX_ENTRY(matrices, CENTER_VERTEX_INDEX, 1, 1) * rhsVec[1] +
        MATRIX_ENTRY(matrices, CENTER_VERTEX_INDEX, 2, 1) * rhsVec[2];
    REAL dz = MATRIX_ENTRY(matrices, CENTER_VERTEX_INDEX, 0, 2) * rhsVec[0] +
        MATRIX_ENTRY(matrices, CENTER_VERTEX_INDEX, 1, 2) * rhsVec[1] +
        MATRIX_ENTRY(matrices, CENTER_VERTEX_INDEX, 2, 2) * rhsVec[2];
    
    dx = dx - vertexToUpdate->x;
    dy = dy - vertexToUpdate->y;
    dz = dz - vertexToUpdate->z;

    residualsOnGPU[residualIndex] = sqrt(dx*dx + dy*dy + dz*dz);
}

__global__ void cuda_updateAllVertexResidualsInBlock(
    Vertex* verticesOnGPU,
    REAL* residualsOnGPU,
    const REAL* __restrict__ matricesOnGPU
) {
    uint3 globalResidualCoord = { 0,0,0 };
    globalResidualCoord.x = blockIdx.x * blockDim.x + threadIdx.x;
    globalResidualCoord.y = blockIdx.y * blockDim.y + threadIdx.y;
    globalResidualCoord.z = blockIdx.z * blockDim.z + threadIdx.z;

    uint3 globalVertexCoord = { globalResidualCoord.x * 2, globalResidualCoord.y * 2, globalResidualCoord.z * 2 };
    const int globalIndex = c_solutionDimensions.y*c_solutionDimensions.x*globalVertexCoord.z + c_solutionDimensions.x*globalVertexCoord.y + globalVertexCoord.x;

    if (!isInsideSolutionSpace(globalVertexCoord)) {
        return;
    }

    Vertex* __restrict__ vertexToUpdate = &verticesOnGPU[globalIndex];

    if (vertexToUpdate->materialConfigId != EMPTY_MATERIALS_CONFIG) {
        REAL rhsVec[3] = { 0,0,0 };
        const REAL* __restrict__ matrices = getPointerToMatricesForVertexGlobalResid(vertexToUpdate, matricesOnGPU);

        buildRHSVectorForVertex(verticesOnGPU, rhsVec, matrices, globalVertexCoord);
        updateVertexResidual(residualsOnGPU, rhsVec, matrices, globalResidualCoord, vertexToUpdate);
    }
}

__host__
extern "C" void cudaLaunchFullResidualUpdateKernel(
    Vertex* verticesOnGPU, 
    REAL* residualsLevelZeroOnGPU,
    REAL* matConfigEquationsOnGPU, 
    const uint3 solutionDims,
    const uint3 residualDims
) {
    dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);
    dim3 blocksPerGrid(residualDims.x / BLOCK_SIZE + 1, residualDims.y / BLOCK_SIZE + 1, residualDims.z / BLOCK_SIZE + 1);

    cudaMemcpyToSymbol(c_solutionDimensions, &solutionDims, sizeof(uint3));
    cudaMemcpyToSymbol(c_residualDimensions, &residualDims, sizeof(uint3));

    cuda_updateAllVertexResidualsInBlock << < blocksPerGrid, threadsPerBlock >> >(verticesOnGPU, residualsLevelZeroOnGPU, matConfigEquationsOnGPU);
    cudaDeviceSynchronize();
    cudaCheckExecution();

}

