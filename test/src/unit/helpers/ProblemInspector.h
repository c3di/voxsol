#pragma once
#include "problem/DiscreteProblem.h"

class ProblemFragment;
class MaterialDictionary;

class ProblemInspector : public DiscreteProblem {
public:
    ProblemInspector(libmmv::Vec3ui size, libmmv::Vec3d voxelSize, MaterialDictionary* matDictionary);

    DirichletBoundary getBoundaryConditionAtVertex(libmmv::Vec3ui& vertex);
};

