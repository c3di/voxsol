#include "stdafx.h"
#include "VTKVisualizer.h"

using namespace std;

VTKVisualizer::VTKVisualizer(Solution* sol) :
    solution(sol),
    numberOfCells(sol->getProblem()->getNumberOfVoxels()),
    numberOfVertices(static_cast<unsigned int>(sol->getVertices()->size()))
{

}

VTKVisualizer::~VTKVisualizer() {

}

void VTKVisualizer::writeToFile(const string& filename) {
    outFile.open(filename, ios::out);

    writeHeader();
    writePositions();
    writeCells();
    writeCellTypes();
    writeCellData();
    writePointData();

    outFile.close();
}

void VTKVisualizer::writeHeader() {
    outFile << "# vtk DataFile Version 2.0" << endl;
    outFile << "Stochastic Mechanical Solver Debug Output" << endl;
    outFile << "ASCII" << endl;
    outFile << "DATASET UNSTRUCTURED_GRID" << endl << endl;
}

void VTKVisualizer::writePositions() {
    outFile << "POINTS " << numberOfVertices << " float" << endl;

    for (unsigned int i = 0; i < numberOfVertices; i++) {
        ettention::Vec3<REAL> pos = solution->getProblem()->getVertexPosition(i);
        outFile << pos.x << " " << pos.y << " " << pos.z << " " << endl;
    }
    
    outFile << endl;
}

void VTKVisualizer::writeCells() {
    const DiscreteProblem* problem = solution->getProblem();

    outFile << "CELLS " << numberOfCells << " " << numberOfCells * 9 << endl;
    for (unsigned int zi = 0; zi < problem->getSize().z; zi++) {
        for (unsigned int yi = 0; yi < problem->getSize().y; yi++) {
            for (unsigned int xi = 0; xi < problem->getSize().x; xi++) {
                writeCell(VoxelCoordinate(xi, yi, zi));
            }
        }
    }
    outFile << endl;
}

void VTKVisualizer::writeCell(VoxelCoordinate& coord) {
    outFile << "8 ";
    writeVertexToCell(0, 0, 0, coord);
    writeVertexToCell(1, 0, 0, coord);
    writeVertexToCell(1, 1, 0, coord);
    writeVertexToCell(0, 1, 0, coord);
    writeVertexToCell(0, 0, 1, coord);
    writeVertexToCell(1, 0, 1, coord);
    writeVertexToCell(1, 1, 1, coord);
    writeVertexToCell(0, 1, 1, coord);
    outFile << endl;
}

void VTKVisualizer::writeVertexToCell(unsigned int xi, unsigned int yi, unsigned int zi, VoxelCoordinate& coord) {
    VertexCoordinate corner = coord + VoxelCoordinate(xi, yi, zi);
    int flatIndexOfCorner = solution->getProblem()->mapToVertexIndex(corner);
    outFile << flatIndexOfCorner << " ";
}

void VTKVisualizer::writeCellTypes() {
    outFile << "CELL_TYPES " << numberOfCells << endl;
    for (unsigned int i = 0; i < numberOfCells; i++) {
        outFile << "12" << endl;
    }
    outFile << endl;
}

void VTKVisualizer::writeCellData() {
    outFile << "CELL_DATA " << numberOfCells << endl;
    writeMaterials();
}

void VTKVisualizer::writeMaterials() {
    outFile << "SCALARS material_id int 1" << endl;
    outFile << "LOOKUP_TABLE default" << endl;
    
    DiscreteProblem* problem = solution->getProblem();
    for (unsigned int i = 0; i < numberOfCells; i++) {
        outFile << static_cast<int>(problem->getMaterial(i)->id) << endl;
    }
    outFile << endl;
}

void VTKVisualizer::writePointData() {
    outFile << "POINT_DATA " << numberOfVertices << endl;

    writeDisplacements();
    writeBoundaries();
}

void VTKVisualizer::writeDisplacements() {
    outFile << "VECTORS displacement float" << endl;

    std::vector<Vertex>* vertices = solution->getVertices();
    for (unsigned int i = 0; i < numberOfVertices; i++) {
        Vertex v = vertices->at(i);
        outFile << v.x << " " << v.y << " " << v.z << endl;
    }
    outFile << endl;
}

void VTKVisualizer::writeBoundaries() {
    DiscreteProblem* problem = solution->getProblem();
    
    outFile << "VECTORS dirichlet_border float" << endl;
    for (unsigned int i = 0; i < numberOfVertices; i++) {
        DirichletBoundary boundary = problem->getDirichletBoundaryAtVertex(i);
        outFile << (boundary.isXFixed() ? 1 : 0) << " " << (boundary.isYFixed() ? 1 : 0) << " " << (boundary.isZFixed() ? 1 : 0) << endl;
    }
    outFile << endl;

    outFile << "VECTORS neumann_border float" << endl;
    for (unsigned int i = 0; i < numberOfVertices; i++) {
        NeumannBoundary boundary = problem->getNeumannBoundaryAtVertex(i);
        outFile << boundary.stress.x << " " << boundary.stress.y << " " << boundary.stress.z << endl;
    }
    outFile << endl;
}
