#pragma once
#include "libmmv/math/Vec3.h"
#include "problem/ProblemFragment.h"
#include "problem/DirichletBoundary.h"
#include "problem/NeumannBoundary.h"
#include "material/Material.h"
#include <vector>
#include <unordered_map>

typedef ettention::Vec3ui VoxelCoordinate;
typedef ettention::Vec3ui VertexCoordinate;

class MaterialDictionary;

class DiscreteProblem {

public:
    DiscreteProblem(ettention::Vec3ui size, ettention::Vec3d voxelSize, MaterialDictionary* matDictionary);
    ~DiscreteProblem();

    void setMaterial(VoxelCoordinate& coordinate, unsigned char matId);
    void setMaterial(unsigned int index, unsigned char matId);
    void setDirichletBoundaryAtVertex(VertexCoordinate& coordinate, DirichletBoundary& condition);
    void setDirichletBoundaryAtVertex(unsigned int index, DirichletBoundary& condition);
    void setNeumannBoundaryAtVertex(VertexCoordinate& coordinate, NeumannBoundary& condition);
    void setNeumannBoundaryAtVertex(unsigned int index, NeumannBoundary& condition);

    Material* getMaterial(VoxelCoordinate& coordinate) const;
    Material* getMaterial(unsigned int index) const;
    ettention::Vec3d getVoxelSize() const;
    ettention::Vec3ui getSize() const;
    unsigned int getNumberOfVoxels() const;
    std::vector<unsigned char>* getMaterialIdVector();
    DirichletBoundary getDirichletBoundaryAtVertex(VertexCoordinate& coordinate);
    DirichletBoundary getDirichletBoundaryAtVertex(unsigned int index);
    NeumannBoundary getNeumannBoundaryAtVertex(VertexCoordinate& coordinate);
    NeumannBoundary getNeumannBoundaryAtVertex(unsigned int index);

    unsigned int mapToVoxelIndex(VoxelCoordinate& coordinate) const;
    VoxelCoordinate mapToVoxelCoordinate(unsigned int index) const;
    unsigned int mapToVertexIndex(VertexCoordinate& coordinate) const;
    VertexCoordinate mapToVertexCoordinate(unsigned int index) const;
    ProblemFragment extractLocalProblem(ettention::Vec3ui centerCoord) const;

    ettention::Vec3<REAL> getVertexPosition(unsigned int index) const;
    ettention::Vec3<REAL> getVertexPosition(VertexCoordinate& coordinate) const;

protected:
    const ettention::Vec3ui problemSize;
    const ettention::Vec3ui solutionSize;
    const ettention::Vec3d voxelSize;
    const unsigned int numberOfCells;
    std::unordered_map<unsigned int, DirichletBoundary> dirichletBoundaryConditions;
    std::unordered_map<unsigned int, NeumannBoundary> neumannBoundaryConditions;
    std::vector<unsigned char> materialIds;
    MaterialDictionary* materialDictionary;

    bool outOfVoxelBounds(VoxelCoordinate& coordinate) const;
    bool outOfVertexBounds(VertexCoordinate& coordinate) const;
    void considerDirichletBoundaryAtLocalProblem(ProblemFragment& fragment) const;
    void considerNeumannBoundaryAtLocalProblem(ProblemFragment& fragment) const;
};
