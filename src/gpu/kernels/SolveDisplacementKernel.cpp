#include "stdafx.h"
#include "SolveDisplacementKernel.h"
#include "problem/boundaryconditions/NeumannBoundary.h"
#include "gpu/GPUParameters.h"
#include <iostream>
#include <fstream>

SolveDisplacementKernel::SolveDisplacementKernel(Solution* sol, BlockSampler* sampler, ResidualVolume* resVol) :
    solution(sol),
    sampler(sampler),
    serializedMatConfigEquations(nullptr),
    serializedVertices(nullptr),
    blockOrigins(nullptr),
    residualVolume(resVol),
    fullResidualUpdateKernel(sol, resVol, nullptr, nullptr)
{
    solutionDimensions.x = sol->getSize().x;
    solutionDimensions.y = sol->getSize().y;
    solutionDimensions.z = sol->getSize().z;

    // Need enough blocks to cover the problem (with room to spare) but not more than the max
    numBlockOriginsPerIteration = std::min( 2 * (1 + solutionDimensions.x / BLOCK_SIZE) * (1 + solutionDimensions.y / BLOCK_SIZE) * (1 + solutionDimensions.z / BLOCK_SIZE), (unsigned int)MAX_BLOCKS_PER_ITERATION);
}

SolveDisplacementKernel::~SolveDisplacementKernel() {
    freeCudaResources();
    assert(serializedMatConfigEquations == nullptr);
    assert(serializedVertices == nullptr);
    assert(blockOrigins == nullptr);
}


void SolveDisplacementKernel::launch() {
    if (!canExecute()) {
        prepareInputs();
    }

    if (canExecute()) {

        sampler->generateNextBlockOrigins(blockOrigins, numBlockOriginsPerIteration);

        cudaLaunchSolveDisplacementKernel(
            serializedVertices,
            serializedMatConfigEquations,
            blockOrigins,
            numBlockOriginsPerIteration,
            solutionDimensions
        );

        numLaunchesSinceLastFullResidualUpdate++;

        if (numLaunchesSinceLastFullResidualUpdate >= numLaunchesBeforeResidualUpdate) {
            fullResidualUpdateKernel.launch();
            numLaunchesSinceLastFullResidualUpdate = 0;
        }
    }
}

void SolveDisplacementKernel::forceResidualUpdate() {
    if (!canExecute()) {
        prepareInputs();
    }

    fullResidualUpdateKernel.launch();
}

void SolveDisplacementKernel::setNumLaunchesBeforeResidualUpdate(unsigned int numLaunches) {
    numLaunchesBeforeResidualUpdate = numLaunches;
}

bool SolveDisplacementKernel::canExecute() {
    if (serializedMatConfigEquations == nullptr || serializedVertices == nullptr || blockOrigins == nullptr) {
        return false;
    }

    return true;
}

void SolveDisplacementKernel::freeCudaResources() {
    if (serializedMatConfigEquations != nullptr) {
        cudaCheckSuccess(cudaFree(serializedMatConfigEquations));
        serializedMatConfigEquations = nullptr;
        fullResidualUpdateKernel.setMatConfigEquationsOnGPU(nullptr);
    }
    if (serializedVertices != nullptr) {
        cudaCheckSuccess(cudaFree(serializedVertices));
        serializedVertices = nullptr;
        fullResidualUpdateKernel.setVerticesOnGPU(nullptr);
    }
    if (blockOrigins != nullptr) {
        cudaCheckSuccess(cudaFree(blockOrigins));
        blockOrigins = nullptr;
    }
}

void SolveDisplacementKernel::prepareInputs() {
    pushMatConfigEquationsManaged();
    pushVerticesManaged();
    allocateBlockOrigins();
}

void SolveDisplacementKernel::allocateBlockOrigins() {
    size_t size = numBlockOriginsPerIteration * sizeof(int3);
    cudaCheckSuccess(cudaMallocManaged(&blockOrigins, size));
}

void SolveDisplacementKernel::pushMatConfigEquationsManaged() {
    size_t size = solution->getMaterialConfigurationEquations()->size() * MaterialConfigurationEquations::SizeInBytes;
    if (size == 0) {
        std::cerr << "ERR: Material configuration equations were not generated before launching the solve displacement kernel!\n";
    }
    cudaCheckSuccess(cudaMallocManaged(&serializedMatConfigEquations, size));
    serializeMaterialConfigurationEquations(serializedMatConfigEquations);
    fullResidualUpdateKernel.setMatConfigEquationsOnGPU(serializedMatConfigEquations);
}

void SolveDisplacementKernel::pushVerticesManaged() {
    const std::vector<Vertex>* vertices = solution->getVertices();
    size_t size = vertices->size() * sizeof(Vertex);
    cudaCheckSuccess(cudaMallocManaged(&serializedVertices, size));
    memcpy(serializedVertices, vertices->data(), size);
    fullResidualUpdateKernel.setVerticesOnGPU(serializedVertices);
}

void SolveDisplacementKernel::pullVertices() {
    std::vector<Vertex>* vertices = solution->getVertices();
    size_t size = vertices->size() * sizeof(Vertex);
    memcpy(vertices->data(), serializedVertices, size);
}

void SolveDisplacementKernel::serializeMaterialConfigurationEquations(void* destination) {
    const std::vector<MaterialConfigurationEquations>* signatures = solution->getMaterialConfigurationEquations();
    size_t size = MaterialConfigurationEquations::SizeInBytes * signatures->size();

    char* serializationPointer = (char*)destination;
    for (unsigned int i = 0; i < signatures->size(); i++) {
        signatures->at(i).serialize(serializationPointer);
        serializationPointer += MaterialConfigurationEquations::SizeInBytes;
    }
}

void SolveDisplacementKernel::solveCPU() {
    for (int iterations = 0; iterations < 1; iterations++) {
        cpuSolveIteration();
    }
}

void SolveDisplacementKernel::cpuBuildRHSVector(libmmv::Vec3<REAL>* rhsVec, const MaterialConfigurationEquations* matrices, int x, int y, int z) {
    int localNeighborIndex;
    int globalNeighborIndex;
    std::vector<Vertex>* vertices = solution->getVertices();
    Vertex zero;

    for (char offsetZ = 0; offsetZ <= 2; offsetZ++) {
        for (char offsetY = 0; offsetY <= 2; offsetY++) {
            for (char offsetX = 0; offsetX <= 2; offsetX++) {

                if (offsetZ == 1 && offsetY == 1 && offsetX == 1) {
                    //Center vertex, skip this one
                    continue;
                }

                int ox = x + offsetX - 1;
                int oy = y + offsetY - 1;
                int oz = z + offsetZ - 1;

                //Local problem size is always 3x3x3 vertices
                localNeighborIndex = offsetZ * 9 + offsetY * 3 + offsetX;
                globalNeighborIndex = oz * solutionDimensions.x*solutionDimensions.y + oy * solutionDimensions.x + ox;

                //vertices outside the solution space contribute nothing, so we can skip them
                const Vertex* neighbor = &zero;
#pragma warning(push)
#pragma warning(disable:4018)
                if (ox >= 0 && ox < solutionDimensions.x && oy >= 0 && oy < solutionDimensions.y && oz >= 0 && oz < solutionDimensions.z) {
                    neighbor = &vertices->at(globalNeighborIndex);
                }
#pragma warning(pop)
                REAL nx = neighbor->x;
                REAL ny = neighbor->y;
                REAL nz = neighbor->z;
                const Matrix3x3* neighborRHS = matrices->getRHS(localNeighborIndex);

                rhsVec->x +=
                    neighborRHS->at(0, 0) * nx +
                    neighborRHS->at(1, 0) * ny +
                    neighborRHS->at(2, 0) * nz;

                rhsVec->y +=
                    neighborRHS->at(0, 1) * nx +
                    neighborRHS->at(1, 1) * ny +
                    neighborRHS->at(2, 1) * nz;

                rhsVec->z +=
                    neighborRHS->at(0, 2) * nx +
                    neighborRHS->at(1, 2) * ny +
                    neighborRHS->at(2, 2) * nz;
            }
        }
    }
}

void SolveDisplacementKernel::cpuSolveIteration() {
    std::vector<Vertex>* vertices = solution->getVertices();
    int flatIndex = -1;
    for (unsigned int z = 0; z < solutionDimensions.z; z++) {
        for (unsigned int y = 0; y < solutionDimensions.y; y++) {
            for (unsigned int x = 0; x < solutionDimensions.x; x++) {
                flatIndex++;
                Vertex* currentVertex = &vertices->at(flatIndex);
                if (currentVertex->materialConfigId == EMPTY_MATERIALS_CONFIG) {
                    continue;
                }
                const MaterialConfigurationEquations* matrices = &solution->getMaterialConfigurationEquations()->at(currentVertex->materialConfigId);
                const NeumannBoundary* neumann = matrices->getNeumannBoundaryCondition();
                const Matrix3x3* lhsInverse = matrices->getLHSInverse();

                // Multiply rhs matrices with displacements of the 26 neighboring vertices
                libmmv::Vec3<REAL> rhsVec(0, 0, 0);
                cpuBuildRHSVector(&rhsVec, matrices, x, y, z);

                // Move to right hand side of equation and add neumann stress
                rhsVec = -rhsVec + neumann->force;

                // Multiply with inverse LHS matrix to get new displacement of center vertex
                currentVertex->x =
                    lhsInverse->at(0, 0) * rhsVec.x +
                    lhsInverse->at(0, 1) * rhsVec.y +
                    lhsInverse->at(0, 2) * rhsVec.z;

                currentVertex->y =
                    lhsInverse->at(1, 0) * rhsVec.x +
                    lhsInverse->at(1, 1) * rhsVec.y +
                    lhsInverse->at(1, 2) * rhsVec.z;

                currentVertex->z =
                    lhsInverse->at(2, 0) * rhsVec.x +
                    lhsInverse->at(2, 1) * rhsVec.y +
                    lhsInverse->at(2, 2) * rhsVec.z;

            }
        }
    }
}

void SolveDisplacementKernel::debugOutputEquationsCPU() {
    std::ofstream outFile;
    //output to %build_folder%/frontend
    outFile.open("matrices_cpu.txt");

    const std::vector<MaterialConfigurationEquations>* equations = solution->getMaterialConfigurationEquations();

    for (int i = 0; i < equations->size(); i++) {
        outFile << "--- " << i << std::endl;

        const MaterialConfigurationEquations* matrices = &equations->at(i);
        const Matrix3x3* mat;

        for (int m = 0; m < 27; m++) {
            if (m == 13) {
                mat = matrices->getLHSInverse();
            }
            else {
                mat = matrices->getRHS(m);
            }
            for (int c = 0; c < 3; c++) {
                for (int r = 0; r < 9; r++) {
                    outFile << mat->at(c,r) << " ";
                }
            }
            outFile << std::endl;
        }
        libmmv::Vec3<REAL> neumann = matrices->getNeumannBoundaryCondition()->force;
        outFile << "Neumann: " << neumann.x << " " << neumann.y << " " << neumann.z << std::endl << std::endl;
    }

    outFile.close();
}

void SolveDisplacementKernel::debugOutputEquationsGPU() {
    std::ofstream outFile;
    //output to %build_folder%/frontend
    outFile.open("matrices_gpu.txt");

    const std::vector<MaterialConfigurationEquations>* equations = solution->getMaterialConfigurationEquations();
    REAL* ptr = serializedMatConfigEquations;

    for (int i = 0; i < equations->size(); i++) {
        outFile << "--- " << i << std::endl;

        for (int m = 0; m < 27; m++) {
            for (int v = 0; v < 9; v++) {
                outFile << *ptr++ << " ";
            }
            outFile << std::endl;
        }
        outFile << "Neumann: " << *ptr++ << " " << *ptr++ << " " << *ptr++ << std::endl << std::endl;
    }

    outFile.close();

}

int3* SolveDisplacementKernel::debugGetImportanceSamplesManaged() {
    return blockOrigins;
}
