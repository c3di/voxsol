#pragma once
#include <libmmv/math/Vec3.h>
#include "Vertex.h"
#include <unordered_map>
#include <vector>
#include <problem\DiscreteProblem.h>
#include <solution/Solution.h>


class SolutionAnalyzer {
public:
    SolutionAnalyzer(DiscreteProblem* problem, Solution* solution);
    ~SolutionAnalyzer();

    REAL* getStrainTensorAt(VoxelCoordinate coord);
    REAL* getStressTensorAt(VoxelCoordinate coord);

    REAL getVonMisesStrainAt(VoxelCoordinate coord);
    REAL getVonMisesStressAt(VoxelCoordinate coord);

    REAL* getStrainTensorAt(int index);
    REAL* getStressTensorAt(int index);

    REAL getVonMisesStrainAt(int index);
    REAL getVonMisesStressAt(int index);

private:
    REAL* voigtNotationStrainTensors;
    REAL* voigtNotationStressTensors;
    REAL* vonMisesStress;
    REAL* vonMisesStrain;
    DiscreteProblem* problem;
    Solution* solution;

    void calculateMechanicalTensors();
    void calculateVonMisesValues();

    void extractNodesForVoxel(std::vector<Vertex>& nodes, unsigned int x, unsigned int y, unsigned int z);

    //void calculatePrincipleStressAndStrain(); 

};