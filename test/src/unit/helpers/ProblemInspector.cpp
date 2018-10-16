#include "helpers/ProblemInspector.h"
#include "material/MaterialDictionary.h"
#include "problem/DiscreteProblem.h"
#include <sstream>

ProblemInspector::ProblemInspector(libmmv::Vec3ui size, libmmv::Vec3d voxelSize, MaterialDictionary* matDictionary) :
    DiscreteProblem(size, voxelSize, matDictionary)
{

}

DirichletBoundary ProblemInspector::getBoundaryConditionAtVertex(libmmv::Vec3ui& vertex) {
    unsigned int index = mapToVertexIndex(vertex);
    return dirichletBoundaryConditions[index];
}