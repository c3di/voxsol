#include "stdafx.h"
#include "LODGenerator.h"

LODGenerator::LODGenerator()
{
}

LODGenerator::~LODGenerator()
{
}

void LODGenerator::populateCoarserLevelProblem(DiscreteProblem* coarseProblem, DiscreteProblem* fineProblem) {
    extrapolateMaterialsToCoarserProblem(fineProblem, coarseProblem);
    extrapolateNeumannBoundariesToCoarserProblem(fineProblem, coarseProblem);
    extrapolateDirichletBoundariesToCoarserProblem(fineProblem, coarseProblem);
}

void LODGenerator::populateCoarserLevelSolution(Solution* coarseSolution, DiscreteProblem* coarseProblem, Solution* fineSolution) {
    libmmv::Vec3ui coarseSize = coarseSolution->getSize();
    std::vector<Vertex>* fineVertices = fineSolution->getVertices();
    std::vector<Vertex>* coarseVertices = coarseSolution->getVertices();

    for (unsigned int z = 0; z < coarseSize.z; z++) {
        for (unsigned int y = 0; y < coarseSize.y; y++) {
            for (unsigned int x = 0; x < coarseSize.x; x++) {
                VertexCoordinate coarseCoord(x, y, z);
                unsigned int fineIndex = fineSolution->mapToIndex(coarseCoord * 2);
                unsigned int coarseIndex = coarseSolution->mapToIndex(coarseCoord);
                Vertex vertex = fineVertices->at(fineIndex);
                
                Vertex* coarseVertex = &coarseVertices->at(coarseIndex);
                coarseVertex->materialConfigId = vertex.materialConfigId;
                coarseVertex->x = vertex.x;
                coarseVertex->y = vertex.y;
                coarseVertex->z = vertex.z;
            }
        }
    }
}

void LODGenerator::projectDisplacementsToFinerLevel(Solution* coarseSolution, Solution* fineSolution) {
    libmmv::Vec3ui fineSize = fineSolution->getSize();
    std::vector<Vertex>* fineVertices = fineSolution->getVertices();
    std::vector<Vertex>* coarseVertices = coarseSolution->getVertices();

    for (int fineIndex = 0; fineIndex < fineVertices->size(); fineIndex++) {
        VertexCoordinate fineCoord = fineSolution->mapToCoordinate(fineIndex);
        VertexCoordinate coarseCoord = fineCoord / 2;
        unsigned int coarseIndex = coarseSolution->mapToIndex(coarseCoord);

        Vertex* fineVertex = &fineVertices->at(fineIndex);

        if (fineVertex->materialConfigId == EMPTY_MATERIALS_CONFIG) {
            // Don't project displacements onto void vertices (empty materials surrounding)
            continue;
        }

        if (existsInCoarserLOD(fineCoord, coarseSolution->getSize())) {
            Vertex* coarseVertex = &coarseVertices->at(coarseIndex);
            fineVertex->x = coarseVertex->x;
            fineVertex->y = coarseVertex->y;
            fineVertex->z = coarseVertex->z;
        }
        else {
            libmmv::Vec3<REAL> interpolatedDisp = interpolateDisplacement(fineCoord, coarseSolution);
            fineVertex->x = interpolatedDisp.x;
            fineVertex->y = interpolatedDisp.y;
            fineVertex->z = interpolatedDisp.z;
        }
    }
}

unsigned char LODGenerator::mergeMaterialsByMode(DiscreteProblem* higherLevel, VoxelCoordinate& fineCoord) {
    std::vector<Material*> materials(8);
    materials[0] = higherLevel->getMaterial(fineCoord);
    materials[1] = higherLevel->getMaterial(fineCoord + VoxelCoordinate(0, 0, 1));
    materials[2] = higherLevel->getMaterial(fineCoord + VoxelCoordinate(0, 1, 0));
    materials[3] = higherLevel->getMaterial(fineCoord + VoxelCoordinate(0, 1, 1));
    materials[4] = higherLevel->getMaterial(fineCoord + VoxelCoordinate(1, 0, 0));
    materials[5] = higherLevel->getMaterial(fineCoord + VoxelCoordinate(1, 0, 1));
    materials[6] = higherLevel->getMaterial(fineCoord + VoxelCoordinate(1, 1, 0));
    materials[7] = higherLevel->getMaterial(fineCoord + VoxelCoordinate(1, 1, 1));

    // Sort in descending order, material 0 is 'empty'
    std::sort(materials.begin(), materials.end(), [](Material* & a, Material* & b) -> bool
    {
        return a->id > b->id;
    });
    int dominantCount = -1;
    int currentCount = 1;
    int currentMaterial = materials[0]->id;
    unsigned char dominantMaterial = materials[0]->id;
    for (int i = 1; i < 8; i++) {
        if (materials[i]->id == 0) {
            if (currentCount > dominantCount) {
                dominantMaterial = currentMaterial;
            }
            break;
        }
        if (materials[i]->id != currentMaterial) {
            if (currentCount > dominantCount) {
                dominantMaterial = currentMaterial;
                dominantCount = currentCount;
            }
            currentMaterial = materials[i]->id;
            currentCount = 0;
        }
        currentCount++;
    }
    
    return dominantMaterial;
}

void LODGenerator::extrapolateMaterialsToCoarserProblem(DiscreteProblem* fineProblem, DiscreteProblem* coarseProblem) {
    libmmv::Vec3ui coarseSize = coarseProblem->getSize();

    for (unsigned int z = 0; z < coarseSize.z; z++) {
        for (unsigned int y = 0; y < coarseSize.y; y++) {
            for (unsigned int x = 0; x < coarseSize.x; x++) {
                VoxelCoordinate coarseCoord(x, y, z);
                VoxelCoordinate fineCoord = coarseCoord * 2;
                unsigned char dominantMaterialId = mergeMaterialsByMode(fineProblem, fineCoord);
                coarseProblem->setMaterial(coarseCoord, dominantMaterialId);
            }
        }
    }
}

void LODGenerator::extrapolateDirichletBoundariesToCoarserProblem(DiscreteProblem* fineProblem, DiscreteProblem* coarseProblem) {
    libmmv::Vec3ui coarseVertexSize = coarseProblem->getSize() + libmmv::Vec3ui(1, 1, 1);
    std::unordered_map<unsigned int, DirichletBoundary>* fineDirichletBoundaries = fineProblem->getDirichletBoundaryMap();
    for (auto it = fineDirichletBoundaries->begin(); it != fineDirichletBoundaries->end(); it++) {
        unsigned int fineIndex = it->first;
        DirichletBoundary fineBoundary = it->second;
        VertexCoordinate fineCoord = fineProblem->mapToVertexCoordinate(fineIndex);
        DirichletBoundary coarseBoundary(fineBoundary);
        
        if (existsInCoarserLOD(fineCoord, coarseVertexSize)) {
            coarseProblem->setDirichletBoundaryAtVertex(fineCoord / 2, coarseBoundary);
        }
        else {
            distributeDirichletBoundaryToNeighbors(coarseBoundary, fineCoord, coarseProblem);
        }
    }
}

void LODGenerator::extrapolateNeumannBoundariesToCoarserProblem(DiscreteProblem* fineProblem, DiscreteProblem* coarseProblem) {
    libmmv::Vec3ui coarseVertexSize = coarseProblem->getSize() + libmmv::Vec3ui(1, 1, 1);
    std::unordered_map<unsigned int, NeumannBoundary>* fineNeumannBoundaries = fineProblem->getNeumannBoundaryMap();
    for (auto it = fineNeumannBoundaries->begin(); it != fineNeumannBoundaries->end(); it++) {
        unsigned int fineIndex = it->first;
        NeumannBoundary fineBoundary = it->second;
        VertexCoordinate fineCoord = fineProblem->mapToVertexCoordinate(fineIndex);
        NeumannBoundary coarseBoundary(fineBoundary);
        
        if (existsInCoarserLOD(fineCoord, coarseVertexSize)) {
            coarseProblem->setNeumannBoundaryAtVertex(fineCoord / 2, coarseBoundary, true);
        }
        else {
            distributeNeumannBoundaryToNeighbors(coarseBoundary, fineCoord, coarseProblem);
        }
    }
}

void LODGenerator::distributeNeumannBoundaryToNeighbors(NeumannBoundary& condition, VertexCoordinate& fineCoord, DiscreteProblem* coarseProblem) {
    std::vector<VertexCoordinate> candidates = findCandidateVerticesForBoundaryConditionProjection(fineCoord, coarseProblem);
    libmmv::Vec3<REAL> distributedForce = condition.force / (REAL)candidates.size();

    for (VertexCoordinate candidate : candidates) {
        NeumannBoundary newCondition(distributedForce);
        coarseProblem->setNeumannBoundaryAtVertex(candidate, newCondition, true);
    }
}

void LODGenerator::distributeDirichletBoundaryToNeighbors(DirichletBoundary& condition, VertexCoordinate& fineCoord, DiscreteProblem* coarseProblem) {
    std::vector<VertexCoordinate> candidates = findCandidateVerticesForBoundaryConditionProjection(fineCoord, coarseProblem);

    for (VertexCoordinate candidate : candidates) {
        DirichletBoundary newCondition(condition);
        coarseProblem->setDirichletBoundaryAtVertex(candidate, newCondition);
    }
}

std::vector<VertexCoordinate> LODGenerator::findCandidateVerticesForBoundaryConditionProjection(VertexCoordinate& fineCoord, DiscreteProblem* coarseProblem) {
    std::vector<VertexCoordinate> candidates;
    libmmv::Vec3ui coarseVertexSize = coarseProblem->getSize() + libmmv::Vec3ui(1, 1, 1);
    for (int z = -1; z <= 1; z++) {
        for (int y = -1; y <= 1; y++) {
            for (int x = -1; x <= 1; x++) {
                VertexCoordinate offset(fineCoord.x + x, fineCoord.y + y, fineCoord.z + z);
                if (existsInCoarserLOD(offset, coarseVertexSize)) {
                    candidates.push_back(offset / 2);
                }
            }
        }
    }
    return candidates;
}

bool LODGenerator::existsInCoarserLOD(libmmv::Vec3ui & fineCoord, libmmv::Vec3ui coarseSize) {
    if (fineCoord.x / 2 >= coarseSize.x || fineCoord.y / 2 >= coarseSize.y || fineCoord.z / 2 >= coarseSize.z) {
        // Fine coord is even but may map to a coarse coord that is outside the coarse level
        return false;
    }
    return true;
}

bool LODGenerator::isEvenCoord(libmmv::Vec3ui & coord) {
    return coord.x % 2 == 0 && coord.y % 2 == 0 && coord.z % 2 == 0;
}

libmmv::Vec3<REAL> LODGenerator::interpolateDisplacement(VertexCoordinate& fineCoord, Solution* coarseSolution) {
    libmmv::Vec3<REAL> totalDisp(0,0,0);
    int totalVertices = 0;
    std::vector<Vertex>* vertices = coarseSolution->getVertices();
    for (int z = -1; z <= 1; z++) {
        for (int y = -1; y <= 1; y++) {
            for (int x = -1; x <= 1; x++) {
                VertexCoordinate offset(fineCoord.x + x, fineCoord.y + y, fineCoord.z + z);
                if (!existsInCoarserLOD(offset, coarseSolution->getSize())) {
                    continue;
                }
                offset = offset / 2;
                unsigned int coarseIndex = coarseSolution->mapToIndex(offset);
                Vertex* coarseVertex = &vertices->at(coarseIndex);
                totalDisp.x += coarseVertex->x;
                totalDisp.y += coarseVertex->y;
                totalDisp.z += coarseVertex->z;
                totalVertices++;
            }
        }
    }
    totalDisp = totalDisp / (REAL)totalVertices;
    return totalDisp;
}


