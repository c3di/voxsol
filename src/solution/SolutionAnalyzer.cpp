#include <stdafx.h>
#include "SolutionAnalyzer.h"

SolutionAnalyzer::SolutionAnalyzer(DiscreteProblem* problem, Solution* solution) :
    solution(solution),
    problem(problem)
{
    calculateMechanicalTensors();
    calculateVonMisesValues();
}

SolutionAnalyzer::~SolutionAnalyzer() {
    if (voigtNotationStrainTensors != nullptr) {
        delete[] voigtNotationStrainTensors;
    }
    if (voigtNotationStressTensors != nullptr) {
        delete[] voigtNotationStressTensors;
    }
    if (vonMisesStress != nullptr) {
        delete[] vonMisesStress;
    }
    if (vonMisesStrain != nullptr) {
        delete[] vonMisesStrain;
    }
}

void SolutionAnalyzer::calculateMechanicalTensors() {
    int numVoxels = problem->getNumberOfVoxels();
    voigtNotationStrainTensors = (REAL*)malloc(numVoxels * 6 * sizeof(REAL));
    voigtNotationStressTensors = (REAL*)malloc(numVoxels * 6 * sizeof(REAL));
    libmmv::Vec3ui problemSize = problem->getSize();
    libmmv::Vec3<REAL> voxelSize = problem->getVoxelSize();

    std::vector<Vertex> localVertices(8);
    int tensorIndex = 0;

    for (unsigned int z = 0; z < problemSize.z; z++) {
        for (unsigned int y = 0; y < problemSize.y; y++) {
            for (unsigned int x = 0; x < problemSize.x; x++) {
                extractNodesForVoxel(localVertices, x, y, z);

                //Derivatives for each direction in the voxel
                REAL du0dx = asREAL(0.25) * (localVertices[1].x - localVertices[0].x + localVertices[3].x - localVertices[2].x + 
                    localVertices[5].x - localVertices[4].x + localVertices[7].x - localVertices[6].x) / voxelSize.x;
                REAL du1dy = asREAL(0.25) * (localVertices[2].y - localVertices[0].y + localVertices[3].y - localVertices[1].y +
                    localVertices[6].y - localVertices[4].y + localVertices[7].y - localVertices[5].y) / voxelSize.y;
                REAL du2dz = asREAL(0.25) * (localVertices[4].z - localVertices[0].z + localVertices[5].z - localVertices[1].z +
                    localVertices[6].z - localVertices[2].z + localVertices[7].z - localVertices[3].z) / voxelSize.z;

                REAL du0dy = asREAL(0.25) * (localVertices[2].x - localVertices[0].x + localVertices[3].x - localVertices[1].x +
                    localVertices[6].x - localVertices[4].x + localVertices[7].x - localVertices[5].x) / voxelSize.y;
                REAL du1dx = asREAL(0.25) * (localVertices[1].y - localVertices[0].y + localVertices[3].y - localVertices[2].y +
                    localVertices[5].y - localVertices[4].y + localVertices[7].y - localVertices[6].y) / voxelSize.x;

                REAL du0dz = asREAL(0.25) * (localVertices[4].x - localVertices[0].x + localVertices[5].x - localVertices[1].x +
                    localVertices[6].x - localVertices[2].x + localVertices[7].x - localVertices[3].x) / voxelSize.z;
                REAL du2dx = asREAL(0.25) * (localVertices[1].z - localVertices[0].z + localVertices[3].z - localVertices[2].z +
                    localVertices[5].z - localVertices[4].z + localVertices[7].z - localVertices[6].z) / voxelSize.x;

                REAL du1dz = asREAL(0.25) * (localVertices[4].y - localVertices[0].y + localVertices[5].y - localVertices[1].y +
                    localVertices[6].y - localVertices[2].y + localVertices[7].y - localVertices[3].y) / voxelSize.z;
                REAL du2dy = asREAL(0.25) * (localVertices[2].z - localVertices[0].z + localVertices[3].z - localVertices[1].z +
                    localVertices[6].z - localVertices[4].z + localVertices[7].z - localVertices[5].z) / voxelSize.y;

                //Deviatoric strains
                REAL exx = asREAL(2.0 / 3.0) * du0dx - asREAL(1.0 / 3.0) * du1dy - asREAL(1.0 / 3.0) * du2dz;
                REAL eyy = -asREAL(1.0 / 3.0) * du0dx + asREAL(2.0 / 3.0) * du1dy - asREAL(1.0 / 3.0) * du2dz;
                REAL ezz = -asREAL(1.0 / 3.0) * du0dx - asREAL(1.0 / 3.0) * du1dy + asREAL(2.0 / 3.0) * du2dz;

                //Engineering strains
                REAL gxy = asREAL(2 * 0.5) * (du0dy + du1dx);
                REAL gxz = asREAL(2 * 0.5) * (du0dz + du2dx);
                REAL gyz = asREAL(2 * 0.5) * (du1dz + du2dy);

                //Entries of the upper diagonal of the 3x3 strain tensor, stored in Voigt Notation
                //https://de.wikipedia.org/wiki/Voigtsche_Notation
                voigtNotationStrainTensors[tensorIndex + 0] = exx;    //e11
                voigtNotationStrainTensors[tensorIndex + 1] = eyy;    //e22
                voigtNotationStrainTensors[tensorIndex + 2] = ezz;    //e33

                voigtNotationStrainTensors[tensorIndex + 3] = gyz;    //2 * e23
                voigtNotationStrainTensors[tensorIndex + 4] = gxz;    //2 * e13
                voigtNotationStrainTensors[tensorIndex + 5] = gxy;    //2 * e12

                Material* voxelMaterial = problem->getMaterial(VoxelCoordinate(x, y, z));
                REAL lambda = voxelMaterial->lambda;
                REAL mu = voxelMaterial->mu;

                voigtNotationStressTensors[tensorIndex + 0] = (asREAL(2.0) * mu + lambda) * exx + lambda * eyy + lambda * ezz;    //sigma11
                voigtNotationStressTensors[tensorIndex + 1] = lambda * exx + (asREAL(2.0) * mu + lambda) * eyy + lambda * ezz;    //sigma22
                voigtNotationStressTensors[tensorIndex + 2] = lambda * exx + lambda * eyy + (asREAL(2.0) * mu + lambda) * ezz;    //sigma33

                voigtNotationStressTensors[tensorIndex + 3] = mu * gyz;    //sigma23
                voigtNotationStressTensors[tensorIndex + 4] = mu * gxz;    //sigma13
                voigtNotationStressTensors[tensorIndex + 5] = mu * gxy;    //sigma12

                tensorIndex += 6;
            }
        }
    }
}

void SolutionAnalyzer::extractNodesForVoxel(std::vector<Vertex>& localVertices, unsigned int x, unsigned int y, unsigned int z) {
    localVertices[0] = solution->getVertexAt(VertexCoordinate(x, y, z));
    localVertices[1] = solution->getVertexAt(VertexCoordinate(x + 1, y, z));
    localVertices[2] = solution->getVertexAt(VertexCoordinate(x, y + 1, z));
    localVertices[3] = solution->getVertexAt(VertexCoordinate(x + 1, y + 1, z));

    localVertices[4] = solution->getVertexAt(VertexCoordinate(x, y, z + 1));
    localVertices[5] = solution->getVertexAt(VertexCoordinate(x + 1, y, z + 1));
    localVertices[6] = solution->getVertexAt(VertexCoordinate(x, y + 1, z + 1));
    localVertices[7] = solution->getVertexAt(VertexCoordinate(x + 1, y + 1, z + 1));
}

void SolutionAnalyzer::calculateVonMisesValues() {
    int numVoxels = problem->getNumberOfVoxels();
    vonMisesStress = (REAL*)malloc(numVoxels * sizeof(REAL));
    vonMisesStrain = (REAL*)malloc(numVoxels * sizeof(REAL));

    for (int voxel = 0; voxel < numVoxels; voxel++) {
        int voxelTensorIndex = voxel * 6;
        REAL stress_xx = voigtNotationStressTensors[voxelTensorIndex + 0];
        REAL stress_yy = voigtNotationStressTensors[voxelTensorIndex + 1];
        REAL stress_zz = voigtNotationStressTensors[voxelTensorIndex + 2];
        REAL stress_yz = voigtNotationStressTensors[voxelTensorIndex + 3];
        REAL stress_xz = voigtNotationStressTensors[voxelTensorIndex + 4];
        REAL stress_xy = voigtNotationStressTensors[voxelTensorIndex + 5];

        vonMisesStress[voxel] = sqrt(stress_xx*stress_xx + stress_yy * stress_yy + stress_zz * stress_zz - stress_xx * stress_yy - stress_xx * stress_zz - stress_yy * stress_zz +
            asREAL(3.0) * (stress_xy*stress_xy + stress_xz * stress_xz + stress_yz * stress_yz));

        REAL strain_xx = voigtNotationStrainTensors[voxelTensorIndex + 0];
        REAL strain_yy = voigtNotationStrainTensors[voxelTensorIndex + 1];
        REAL strain_zz = voigtNotationStrainTensors[voxelTensorIndex + 2];
        REAL strain_yz = voigtNotationStrainTensors[voxelTensorIndex + 3];
        REAL strain_xz = voigtNotationStrainTensors[voxelTensorIndex + 4];
        REAL strain_xy = voigtNotationStrainTensors[voxelTensorIndex + 5];

        vonMisesStrain[voxel] = asREAL(2.0 / 3.0) * sqrt(asREAL(3.0 / 2.0) * (strain_xx*strain_xx + strain_yy * strain_yy + strain_zz * strain_zz) + 
            asREAL(3.0 / 4.0) * (strain_xy*strain_xy + strain_xz * strain_xz + strain_yz * strain_yz));
    }
}

REAL* SolutionAnalyzer::getStrainTensorAt(VoxelCoordinate coord) {
    int voxelIndex = problem->mapToVoxelIndex(coord);
    return getStrainTensorAt(voxelIndex);
}

REAL* SolutionAnalyzer::getStressTensorAt(VoxelCoordinate coord) {
    int voxelIndex = problem->mapToVoxelIndex(coord);
    return getStressTensorAt(voxelIndex);
}

REAL SolutionAnalyzer::getVonMisesStrainAt(VoxelCoordinate coord) {
    int voxelIndex = problem->mapToVoxelIndex(coord);
    return getVonMisesStrainAt(voxelIndex);
}

REAL SolutionAnalyzer::getVonMisesStressAt(VoxelCoordinate coord) {
    int voxelIndex = problem->mapToVoxelIndex(coord);
    return getVonMisesStressAt(voxelIndex);
}

REAL* SolutionAnalyzer::getStrainTensorAt(int voxelIndex) {
    return &voigtNotationStrainTensors[voxelIndex * 6];
}

REAL* SolutionAnalyzer::getStressTensorAt(int voxelIndex) {
    return &voigtNotationStressTensors[voxelIndex * 6];
}

REAL SolutionAnalyzer::getVonMisesStrainAt(int voxelIndex) {
    return vonMisesStrain[voxelIndex];
}

REAL SolutionAnalyzer::getVonMisesStressAt(int voxelIndex) {
    return vonMisesStress[voxelIndex];
}

