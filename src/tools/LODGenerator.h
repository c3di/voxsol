#pragma once
#include "problem/DiscreteProblem.h"
#include "solution/Solution.h"
#include "gpu/sampling/ResidualVolume.h"

class LODGenerator {
public:

    LODGenerator();
    ~LODGenerator();

    void populateCoarserLevelProblem(DiscreteProblem* coarseProblem, DiscreteProblem* higherLevel);
    void populateCoarserLevelSolution(Solution* coarseSolution, DiscreteProblem* coarseProblem, Solution* fineSolution);

    void projectDisplacementsToFinerLevel(Solution* coarseSolution, Solution* fineSolution);

protected:

    unsigned char mergeMaterialsByMode(DiscreteProblem* higherLevel, VoxelCoordinate& fineCoord);

    void extrapolateMaterialsToCoarserProblem(DiscreteProblem* fineProblem, DiscreteProblem* coarseProblem);
    void extrapolateDirichletBoundariesToCoarserProblem(DiscreteProblem* fineProblem, DiscreteProblem* coarseProblem);
    void extrapolateNeumannBoundariesToCoarserProblem(DiscreteProblem* fineProblem, DiscreteProblem* coarseProblem);

    void distributeNeumannBoundaryToNeighbors(NeumannBoundary& condition, VertexCoordinate& fineCoord, DiscreteProblem* coarseProblem);
    void distributeDirichletBoundaryToNeighbors(DirichletBoundary& condition, VertexCoordinate& fineCoord, DiscreteProblem* coarseProblem);
    std::vector<VertexCoordinate> findCandidateVerticesForBoundaryConditionProjection(VertexCoordinate& fineCoord, DiscreteProblem* coarseProblem);
    bool existsInCoarserLOD(libmmv::Vec3ui& fineCoord, libmmv::Vec3ui coarseSize);
    bool isEvenCoord(libmmv::Vec3ui& coord);
    libmmv::Vec3<REAL> interpolateDisplacement(VertexCoordinate& fineCoord, Solution* coarseSolution);
};
