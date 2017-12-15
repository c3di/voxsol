#include <stdafx.h>
#include "Solution.h"
#include "problem/ProblemFragment.h"
#include "problem/DiscreteProblem.h"
#include "material/MaterialConfigurationEquationsFactory.h"
#include "material/MaterialConfiguration.h"

Solution::Solution(DiscreteProblem& problem) :
    size(problem.getSize() + libmmv::Vec3ui(1,1,1)),
    voxelSize(problem.getVoxelSize()),
    problem(&problem),
    vertices(size.x * size.y * size.z),
    vertexDiff(size.x * size.y * size.z)
{

}

Solution::~Solution() {

}

std::vector<Vertex>* Solution::getVertices() {
    return &vertices;
}

std::vector<Vertex>* Solution::getDifferences() {
    return &vertexDiff;
}

const libmmv::Vec3ui Solution::getSize() const {
    return size;
}

DiscreteProblem* Solution::getProblem() {
    return problem;
}

const std::vector<MaterialConfigurationEquations>* Solution::getMaterialConfigurationEquations() const {
    return &matConfigEquations;
}

unsigned int Solution::mapToIndex(libmmv::Vec3ui& coordinate) const {
    if (outOfBounds(coordinate)) {
        throw std::invalid_argument("given coordinate cannot be mapped to an index because it is outside the solution space");
    }
    return coordinate.x + coordinate.y * size.x + coordinate.z * size.x * size.y;
}

libmmv::Vec3ui Solution::mapToCoordinate(unsigned int index) const {
    return libmmv::Vec3ui(index % size.x, (index / size.x) % size.y, index / (size.x * size.y));
}

bool Solution::outOfBounds(libmmv::Vec3ui& coordinate) const {
    return coordinate.x < 0 || coordinate.x >= size.x || coordinate.y < 0 || coordinate.y >= size.y || coordinate.z < 0 || coordinate.z >= size.z;
}

void Solution::computeMaterialConfigurationEquations() {
    // This is separated into two steps to allow matrix computation to be done asynchronously later
    gatherUniqueMaterialConfigurations();
    computeEquationsForUniqueMaterialConfigurations();
}

void Solution::gatherUniqueMaterialConfigurations() {
    ConfigId equationIdCounter = 0;
    std::unordered_map < MaterialConfiguration, ConfigId> matConfigToEquationId;

    for (unsigned int z = 0; z < size.z; z++) {
        for (unsigned int y = 0; y < size.y; y++) {
            for (unsigned int x = 0; x < size.x; x++) {
                libmmv::Vec3ui centerCoord(x, y, z);
                ProblemFragment fragment = problem->extractLocalProblem(centerCoord);
                MaterialConfiguration materialConfiguration = fragment.getMaterialConfiguration();

                if (matConfigToEquationId.count(materialConfiguration) <= 0) {
                    matConfigToEquationId[materialConfiguration] = equationIdCounter;
                    equationIdCounter++;
                }
                vertices[mapToIndex(centerCoord)].materialConfigId = matConfigToEquationId[materialConfiguration];
            }
        }
    }
    matConfigEquations.resize(equationIdCounter);
    std::cout << "Found " << equationIdCounter << " unique problem configurations\n";
}

void Solution::computeEquationsForUniqueMaterialConfigurations() {
    MaterialConfigurationEquationsFactory mceFactory(voxelSize);

    for (int i = 0; i < vertices.size(); i++) {
        int equationId = vertices[i].materialConfigId;
        MaterialConfigurationEquations* equations = &matConfigEquations[equationId];

        if (!equations->isInitialized()) {
            equations->setId(equationId);
            libmmv::Vec3ui centerCoord = mapToCoordinate(i);
            ProblemFragment fragment = problem->extractLocalProblem(centerCoord);
            mceFactory.initializeEquationsForFragment(equations, fragment);
        }
    }
}

void Solution::updateDisplacements(Vertex* serializedVertices) {
    Vertex* updatedVertex(serializedVertices);
    for (int i = 0; i < vertices.size(); i++) {
        Vertex* vertex = &vertices.at(i);
        Vertex* diff = &vertexDiff.at(i);

        diff->x = vertex->x - updatedVertex->x;
        diff->y = vertex->y - updatedVertex->y;
        diff->z = vertex->z - updatedVertex->z;

        vertex->x = updatedVertex->x;
        vertex->y = updatedVertex->y;
        vertex->z = updatedVertex->z;
        
        updatedVertex++;
    }
}
