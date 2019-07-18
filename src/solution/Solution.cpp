#include <stdafx.h>
#include "Solution.h"
#include <iomanip>
#include "problem/ProblemFragment.h"
#include "problem/DiscreteProblem.h"
#include "material/MaterialConfigurationEquationsFactory.h"
#include "material/MaterialConfiguration.h"


Solution::Solution(DiscreteProblem* problem) :
    size(problem->getSize() + libmmv::Vec3ui(1,1,1)),
    voxelSize(problem->getVoxelSize()),
    problem(problem),
    vertices(size.x * size.y * size.z)
{

}

Solution::~Solution() {

}

std::vector<Vertex>* Solution::getVertices() {
    return &vertices;
}

Vertex Solution::getVertexAt(VertexCoordinate coord)
{
    int index = mapToIndex(coord);
    return vertices.at(index);
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

void Solution::createVoidMaterialConfiguration(std::unordered_map<MaterialConfiguration, UniqueConfig>& matConfigToEquation) {
    std::vector<Material*> emptyMats = {&Material::EMPTY,&Material::EMPTY, &Material::EMPTY, &Material::EMPTY, &Material::EMPTY, &Material::EMPTY, &Material::EMPTY, &Material::EMPTY};
    MaterialConfiguration voidConfig(&emptyMats);
    matConfigToEquation[voidConfig].equationId = 0;
}

void Solution::gatherUniqueMaterialConfigurations() {
    std::unordered_map < MaterialConfiguration, UniqueConfig> matConfigToEquation;

    createVoidMaterialConfiguration(matConfigToEquation);
    scanSolutionForUniqueConfigurations(matConfigToEquation);
    sortUniqueConfigurationsByFrequency(matConfigToEquation);
    assignConfigurationIdsToVertices(matConfigToEquation);
    
    matConfigEquations.resize(matConfigToEquation.size());

    std::cout << "Found " << matConfigToEquation.size() << " unique local problem configurations\n";
}

void Solution::scanSolutionForUniqueConfigurations(std::unordered_map<MaterialConfiguration, UniqueConfig>& matConfigToEquation) {
    ConfigId equationIdCounter = 1; // 0 is reserved for the void material configuration (all empty materials)

    for (unsigned int z = 0; z < size.z; z++) {
        for (unsigned int y = 0; y < size.y; y++) {
            for (unsigned int x = 0; x < size.x; x++) {
                libmmv::Vec3ui centerCoord(x, y, z);
                ProblemFragment fragment = problem->extractLocalProblem(centerCoord);
                MaterialConfiguration materialConfiguration = fragment.getMaterialConfiguration();

                if (matConfigToEquation.count(materialConfiguration) <= 0) {
                    matConfigToEquation[materialConfiguration].equationId = equationIdCounter;
                    equationIdCounter++;
                }

                // Necessary if sorting is disabled
                //Vertex* vertex = &vertices[mapToIndex(centerCoord)];
                //vertex->materialConfigId = matConfigToEquation[materialConfiguration].equationId;

                matConfigToEquation[materialConfiguration].numInstancesInProblem++;
            }
        }
    }
}

void Solution::sortUniqueConfigurationsByFrequency(std::unordered_map<MaterialConfiguration, UniqueConfig>& matConfigToEquation) {
    std::vector<UniqueConfig*> sortedByFrequency;

    for (auto it = matConfigToEquation.begin(); it != matConfigToEquation.end(); it++) {
        sortedByFrequency.push_back(&it->second);

#ifdef OUTPUT_RARE_CONFIGURATIONS_DEBUG
        if (it->second.numInstancesInProblem <= OUTPUT_RARE_CONFIGURATIONS_DEBUG) {
            MaterialConfiguration materialConfiguration = it->first;
            std::stringstream ss;
            ss << "  [Rare conf] Dirichlet: " << materialConfiguration.dirichletBoundaryCondition.fixed;
            ss << std::setprecision(10) << "  Neumann: (" << materialConfiguration.neumannBoundaryCondition.stress.x << "," << materialConfiguration.neumannBoundaryCondition.stress.y << "," << materialConfiguration.neumannBoundaryCondition.stress.z;
            ss << ")  Materials: [";
            for (int i = 0; i < 8; i++) {
                ss << std::setw(3) << static_cast<int>(materialConfiguration.ids[i]) << " ";
            }
            ss << "\b] Hash: " << std::hash<MaterialConfiguration>{}(materialConfiguration);
            ss << "\t Occurs " << matConfigToEquation[materialConfiguration].numInstancesInProblem << " times" << std::endl;
            std::cout << ss.str();
        }
#endif

    }

    std::sort(sortedByFrequency.begin(), sortedByFrequency.end(), [](const UniqueConfig* a, const UniqueConfig* b) -> bool {
        return a->numInstancesInProblem > b->numInstancesInProblem;
    });

    //The sorted position in the array becomes the configuration's ID. This way ID 1 will be the most common config, 2 the second most common etc.
    // ID 0 is reserved for the void material configuration
    for (int i = 0; i < sortedByFrequency.size(); i++) {
        sortedByFrequency[i]->equationId = i+1;
    }
}

void Solution::assignConfigurationIdsToVertices(std::unordered_map<MaterialConfiguration, UniqueConfig>& matConfigToEquation) {
    for (unsigned int z = 0; z < size.z; z++) {
        for (unsigned int y = 0; y < size.y; y++) {
            for (unsigned int x = 0; x < size.x; x++) {
                libmmv::Vec3ui centerCoord(x, y, z);
                ProblemFragment fragment = problem->extractLocalProblem(centerCoord);
                MaterialConfiguration materialConfiguration = fragment.getMaterialConfiguration();

                Vertex* vertex = &vertices[mapToIndex(centerCoord)];
                vertex->materialConfigId = matConfigToEquation[materialConfiguration].equationId;
            }
        }
    }
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

