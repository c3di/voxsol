#pragma once
#include "libmmv/math/Vec3.h"
#include "problem/ProblemFragment.h"
#include "problem/boundaryconditions/DirichletBoundary.h"
#include "problem/boundaryconditions/NeumannBoundary.h"
#include "problem/boundaryconditions/DisplacementBoundary.h"
#include "material/Material.h"
#include <vector>
#include <unordered_map>

typedef libmmv::Vec3ui VoxelCoordinate;
typedef libmmv::Vec3ui VertexCoordinate;

class MaterialDictionary;

class DiscreteProblem {

public:
    DiscreteProblem(libmmv::Vec3ui size, libmmv::Vec3<REAL> voxelSize, MaterialDictionary* matDictionary);
    ~DiscreteProblem();

    void setMaterial(VoxelCoordinate& coordinate, unsigned char matId);
    void setMaterial(unsigned int index, unsigned char matId);
    void setDirichletBoundaryAtVertex(VertexCoordinate& coordinate, DirichletBoundary& condition);
    void setDirichletBoundaryAtVertex(unsigned int index, DirichletBoundary& condition);
    void setNeumannBoundaryAtVertex(VertexCoordinate& coordinate, NeumannBoundary& condition, bool combineIfAlreadyExists = false);
    void setNeumannBoundaryAtVertex(unsigned int index, NeumannBoundary& condition, bool combineIfAlreadyExists = false);
    void setDisplacementBoundaryAtVertex(VertexCoordinate& coordinate, DisplacementBoundary& condition);
    void setDisplacementBoundaryAtVertex(unsigned int index, DisplacementBoundary& condition);

    Material* getMaterial(VoxelCoordinate& coordinate) const;
    Material* getMaterial(unsigned int index) const;
    libmmv::Vec3d getVoxelSize() const;
    libmmv::Vec3ui getSize() const;
    unsigned int getNumberOfVoxels() const;
    std::vector<unsigned char>* getMaterialIdVector();
    DirichletBoundary getDirichletBoundaryAtVertex(VertexCoordinate& coordinate);
    DirichletBoundary getDirichletBoundaryAtVertex(unsigned int index);
    NeumannBoundary getNeumannBoundaryAtVertex(VertexCoordinate& coordinate);
    NeumannBoundary getNeumannBoundaryAtVertex(unsigned int index);
    std::unordered_map<unsigned int, NeumannBoundary>* getNeumannBoundaryMap();
    std::unordered_map<unsigned int, DirichletBoundary>* getDirichletBoundaryMap();
    std::unordered_map<unsigned int, DisplacementBoundary>* getDisplacementBoundaryMap();
    MaterialDictionary* getMaterialDictionary();

    unsigned int mapToVoxelIndex(VoxelCoordinate& coordinate) const;
    VoxelCoordinate mapToVoxelCoordinate(unsigned int index) const;
    unsigned int mapToVertexIndex(VertexCoordinate& coordinate) const;
    VertexCoordinate mapToVertexCoordinate(unsigned int index) const;
    ProblemFragment extractLocalProblem(VertexCoordinate& centerCoord) const;

    libmmv::Vec3<REAL> getVertexPosition(unsigned int index) const;
    libmmv::Vec3<REAL> getVertexPosition(VertexCoordinate& coordinate) const;

protected:
    const libmmv::Vec3ui problemSize;
    const libmmv::Vec3ui solutionSize;
    const libmmv::Vec3<REAL> voxelSize;
    const unsigned int numberOfCells;
    MaterialDictionary* materialDictionary;
    std::unordered_map<unsigned int, DirichletBoundary> dirichletBoundaryConditions;
    std::unordered_map<unsigned int, NeumannBoundary> neumannBoundaryConditions;
    std::unordered_map<unsigned int, DisplacementBoundary> displacementBoundaryConditions;
    std::vector<unsigned char> materialIds;

    bool outOfVoxelBounds(VoxelCoordinate& coordinate) const;
    bool outOfVertexBounds(VertexCoordinate& coordinate) const;
    void considerDirichletBoundaryAtLocalProblem(ProblemFragment& fragment) const;
    void considerNeumannBoundaryAtLocalProblem(ProblemFragment& fragment) const; 
    void considerDisplacementBoundaryAtLocalProblem(ProblemFragment& fragment) const;
};
