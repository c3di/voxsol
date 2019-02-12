#include "stdafx.h"
#include "VTKSolutionVisualizer.h"
#include "solution/SolutionAnalyzer.h"

using namespace std;

VTKSolutionVisualizer::VTKSolutionVisualizer(Solution* sol, ResidualVolume* impVol) :
    solution(sol),
    numberOfCells(sol->getProblem()->getNumberOfVoxels()),
    numberOfVertices(static_cast<unsigned int>(sol->getVertices()->size())),
    impVol(impVol)
{

}

VTKSolutionVisualizer::~VTKSolutionVisualizer() {

}

void VTKSolutionVisualizer::filterOutNullVoxels(bool doFilter) {
    filterNullVoxels = doFilter;
    if (doFilter) {
        std::vector<Vertex>* vertices = solution->getVertices();
        unsigned int numRemainingVertices = 0;
        for (unsigned int i = 0; i < solution->getVertices()->size(); i++) {
            if (vertices->at(i).materialConfigId != 0) {
                vertexOrigToFilteredIndex[i] = numRemainingVertices;
                vertexFilteredToOrigIndex[numRemainingVertices] = i;
                numRemainingVertices++;
            }
        }

        unsigned int numRemainingCells = 0;
        DiscreteProblem* problem = solution->getProblem();
        for (unsigned int i = 0; i < problem->getNumberOfVoxels(); i++) {
            if (problem->getMaterial(i)->id != Material::EMPTY.id) {
                cellOrigToFilteredIndex[i] = numRemainingCells;
                cellFilteredToOrigIndex[numRemainingCells] = i;
                numRemainingCells++;
            }
        }
        numberOfVertices = numRemainingVertices;
        numberOfCells = numRemainingCells;
    }
    else {
        numberOfVertices = static_cast<unsigned int>(solution->getVertices()->size());
    }
}

void VTKSolutionVisualizer::setMechanicalValuesOutput(bool flag) {
    enableMechanicalValuesOutput = flag;
}

void VTKSolutionVisualizer::writeToFile(const string& filename) {
    outFile.open(filename, ios::out);
    std::cout << "Starting VTK output\n";
    writeHeader();
    writePositions();
    std::cout << "Wrote positions\n";
    writeCells();
    writeCellTypes();
    writeCellData();
    std::cout << "Wrote cells\n";
    writePointData();
    std::cout << "Wrote point data\n";

    outFile.close();
}

void VTKSolutionVisualizer::writeHeader() {
    outFile << "# vtk DataFile Version 2.0" << endl;
    outFile << "Stochastic Mechanical Solver Debug Output" << endl;
    outFile << "ASCII" << endl;
    outFile << "DATASET UNSTRUCTURED_GRID" << endl << endl;
}

void VTKSolutionVisualizer::writePositions() {
    outFile << "POINTS " << numberOfVertices << " double" << endl;
    std::vector<Vertex>* vertices = solution->getVertices();
    for (unsigned int i = 0; i < numberOfVertices; i++) {
        unsigned int index = filterNullVoxels ? vertexFilteredToOrigIndex[i] : i;
        if (filterNullVoxels && vertices->at(index).materialConfigId == 0) {
            continue;
        }
        libmmv::Vec3<REAL> pos = solution->getProblem()->getVertexPosition(index);
        outFile << pos.x << " " << pos.y << " " << pos.z << " " << endl;
    }
    
    outFile << endl;
}

void VTKSolutionVisualizer::writeCells() {
    const DiscreteProblem* problem = solution->getProblem();

    outFile << "CELLS " << numberOfCells << " " << numberOfCells * 9 << endl;
    for (unsigned int zi = 0; zi < problem->getSize().z; zi++) {
        for (unsigned int yi = 0; yi < problem->getSize().y; yi++) {
            for (unsigned int xi = 0; xi < problem->getSize().x; xi++) {
                VoxelCoordinate coord(xi, yi, zi);
                unsigned int flatIndex = problem->mapToVoxelIndex(coord);
                if (filterNullVoxels && cellOrigToFilteredIndex.count(flatIndex) <= 0) {
                    // There is no mapping for this voxel index, so it must have been filtered out due to null material
                    continue;
                }
                writeCell(coord);
            }
        }
    }
    outFile << endl;
}

void VTKSolutionVisualizer::writeCell(VoxelCoordinate& coord) {
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

void VTKSolutionVisualizer::writeVertexToCell(unsigned int xi, unsigned int yi, unsigned int zi, VoxelCoordinate& coord) {
    VertexCoordinate corner = coord + VoxelCoordinate(xi, yi, zi);
    int flatIndexOfCorner = solution->getProblem()->mapToVertexIndex(corner);
    if (filterNullVoxels) {
        if (vertexOrigToFilteredIndex.count(flatIndexOfCorner) <= 0) {
            throw std::exception("Could not find mapping for vertex");
        }
        flatIndexOfCorner = vertexOrigToFilteredIndex[flatIndexOfCorner];
    }
    outFile << flatIndexOfCorner << " ";
}

void VTKSolutionVisualizer::writeCellTypes() {
    outFile << "CELL_TYPES " << numberOfCells << endl;
    for (unsigned int i = 0; i < numberOfCells; i++) {
        outFile << "12" << endl;
    }
    outFile << endl;
}

void VTKSolutionVisualizer::writeCellData() {
    outFile << "CELL_DATA " << numberOfCells << endl;

    writeMaterials();

    if (enableMechanicalValuesOutput) {
        std::cout << "Writing Mechanical values..." << std::endl;
        SolutionAnalyzer solutionAnalyzer(solution->getProblem(), solution);
        writeVonMisesStresses(&solutionAnalyzer);
        writeVonMisesStrains(&solutionAnalyzer);
        writeStressTensors(&solutionAnalyzer);
        writeStrainTensors(&solutionAnalyzer);
    }
}

void VTKSolutionVisualizer::writeMaterials() {
    outFile << "SCALARS material_id int 1" << endl;
    outFile << "LOOKUP_TABLE default" << endl;
    
    DiscreteProblem* problem = solution->getProblem();
    for (unsigned int i = 0; i < numberOfCells; i++) {
        unsigned int index = filterNullVoxels ? cellFilteredToOrigIndex[i] : i;
        int matId = static_cast<int>(problem->getMaterial(index)->id);
        outFile << matId << endl;
    }
    outFile << endl;
}

void VTKSolutionVisualizer::writeVonMisesStresses(SolutionAnalyzer* solutionAnalyzer) {
    outFile << "SCALARS von_Mises_stress float 1" << endl;
    outFile << "LOOKUP_TABLE default" << endl;

    for (unsigned int i = 0; i < numberOfCells; i++) {
        unsigned int index = filterNullVoxels ? cellFilteredToOrigIndex[i] : i;
        REAL stress = static_cast<REAL>(solutionAnalyzer->getVonMisesStressAt(index));
        outFile << stress << endl;
    }
    outFile << endl;
}

void VTKSolutionVisualizer::writeVonMisesStrains(SolutionAnalyzer* solutionAnalyzer) {
    outFile << "SCALARS von_Mises_strain float 1" << endl;
    outFile << "LOOKUP_TABLE default" << endl;

    for (unsigned int i = 0; i < numberOfCells; i++) {
        unsigned int index = filterNullVoxels ? cellFilteredToOrigIndex[i] : i;
        REAL stress = static_cast<REAL>(solutionAnalyzer->getVonMisesStrainAt(index));
        outFile << stress << endl;
    }
    outFile << endl;
}

void VTKSolutionVisualizer::writeStressTensors(SolutionAnalyzer* solutionAnalyzer) {
    outFile << "SCALARS stress_sigma_xx float 1" << endl;
    outFile << "LOOKUP_TABLE default" << endl;

    for (unsigned int i = 0; i < numberOfCells; i++) {
        unsigned int index = filterNullVoxels ? cellFilteredToOrigIndex[i] : i;
        REAL* stressTensor = solutionAnalyzer->getStressTensorAt(index);
        outFile << stressTensor[0] << endl;
    }
    outFile << endl;

    outFile << "SCALARS stress_sigma_yy float 1" << endl;
    outFile << "LOOKUP_TABLE default" << endl;

    for (unsigned int i = 0; i < numberOfCells; i++) {
        unsigned int index = filterNullVoxels ? cellFilteredToOrigIndex[i] : i;
        REAL* stressTensor = solutionAnalyzer->getStressTensorAt(index);
        outFile << stressTensor[1] << endl;
    }
    outFile << endl;

    outFile << "SCALARS stress_sigma_zz float 1" << endl;
    outFile << "LOOKUP_TABLE default" << endl;

    for (unsigned int i = 0; i < numberOfCells; i++) {
        unsigned int index = filterNullVoxels ? cellFilteredToOrigIndex[i] : i;
        REAL* stressTensor = solutionAnalyzer->getStressTensorAt(index);
        outFile << stressTensor[2] << endl;
    }
    outFile << endl;

    outFile << "SCALARS stress_tau_yz float 1" << endl;
    outFile << "LOOKUP_TABLE default" << endl;

    for (unsigned int i = 0; i < numberOfCells; i++) {
        unsigned int index = filterNullVoxels ? cellFilteredToOrigIndex[i] : i;
        REAL* stressTensor = solutionAnalyzer->getStressTensorAt(index);
        outFile << stressTensor[3] << endl;
    }
    outFile << endl;
    
    outFile << "SCALARS stress_tau_xz float 1" << endl;
    outFile << "LOOKUP_TABLE default" << endl;

    for (unsigned int i = 0; i < numberOfCells; i++) {
        unsigned int index = filterNullVoxels ? cellFilteredToOrigIndex[i] : i;
        REAL* stressTensor = solutionAnalyzer->getStressTensorAt(index);
        outFile << stressTensor[4] << endl;
    }
    outFile << endl;

    outFile << "SCALARS stress_tau_xy float 1" << endl;
    outFile << "LOOKUP_TABLE default" << endl;

    for (unsigned int i = 0; i < numberOfCells; i++) {
        unsigned int index = filterNullVoxels ? cellFilteredToOrigIndex[i] : i;
        REAL* stressTensor = solutionAnalyzer->getStressTensorAt(index);
        outFile << stressTensor[5] << endl;
    }
    outFile << endl;
}

void VTKSolutionVisualizer::writeStrainTensors(SolutionAnalyzer* solutionAnalyzer) {
    outFile << "SCALARS strain_e_xx float 1" << endl;
    outFile << "LOOKUP_TABLE default" << endl;

    for (unsigned int i = 0; i < numberOfCells; i++) {
        unsigned int index = filterNullVoxels ? cellFilteredToOrigIndex[i] : i;
        REAL* strainTensor = solutionAnalyzer->getStrainTensorAt(index);
        outFile << strainTensor[0] << endl;
    }
    outFile << endl;

    outFile << "SCALARS strain_e_yy float 1" << endl;
    outFile << "LOOKUP_TABLE default" << endl;

    for (unsigned int i = 0; i < numberOfCells; i++) {
        unsigned int index = filterNullVoxels ? cellFilteredToOrigIndex[i] : i;
        REAL* strainTensor = solutionAnalyzer->getStrainTensorAt(index);
        outFile << strainTensor[1] << endl;
    }
    outFile << endl;

    outFile << "SCALARS strain_e_zz float 1" << endl;
    outFile << "LOOKUP_TABLE default" << endl;

    for (unsigned int i = 0; i < numberOfCells; i++) {
        unsigned int index = filterNullVoxels ? cellFilteredToOrigIndex[i] : i;
        REAL* strainTensor = solutionAnalyzer->getStrainTensorAt(index);
        outFile << strainTensor[2] << endl;
    }
    outFile << endl;

    outFile << "SCALARS strain_gamma_yz float 1" << endl;
    outFile << "LOOKUP_TABLE default" << endl;

    for (unsigned int i = 0; i < numberOfCells; i++) {
        unsigned int index = filterNullVoxels ? cellFilteredToOrigIndex[i] : i;
        REAL* strainTensor = solutionAnalyzer->getStrainTensorAt(index);
        outFile << strainTensor[3] << endl;
    }
    outFile << endl;

    outFile << "SCALARS strain_gamma_xz float 1" << endl;
    outFile << "LOOKUP_TABLE default" << endl;

    for (unsigned int i = 0; i < numberOfCells; i++) {
        unsigned int index = filterNullVoxels ? cellFilteredToOrigIndex[i] : i;
        REAL* strainTensor = solutionAnalyzer->getStrainTensorAt(index);
        outFile << strainTensor[4] << endl;
    }
    outFile << endl;

    outFile << "SCALARS strain_gamma_xy float 1" << endl;
    outFile << "LOOKUP_TABLE default" << endl;

    for (unsigned int i = 0; i < numberOfCells; i++) {
        unsigned int index = filterNullVoxels ? cellFilteredToOrigIndex[i] : i;
        REAL* strainTensor = solutionAnalyzer->getStrainTensorAt(index);
        outFile << strainTensor[5] << endl;
    }
    outFile << endl;
}

void VTKSolutionVisualizer::writePointData() {
    outFile << "POINT_DATA " << numberOfVertices << endl;

    writeDisplacements();
    writeBoundaries();

    if (enableResidualOutput && impVol != nullptr) {
        writeResiduals();
    }
    if (enableMatConfigIdOutput) {
        writeMaterialConfigIds();
    }
}

void VTKSolutionVisualizer::writeDisplacements() {
    outFile << "VECTORS displacement double" << endl;

    std::vector<Vertex>* vertices = solution->getVertices();
    for (unsigned int i = 0; i < numberOfVertices; i++) {
        unsigned int index = filterNullVoxels ? vertexFilteredToOrigIndex[i] : i;
        Vertex v = vertices->at(index);
        outFile << v.x << " " << v.y << " " << v.z << endl;
    }
    outFile << endl;
}

void VTKSolutionVisualizer::writeBoundaries() {
    DiscreteProblem* problem = solution->getProblem();
    std::vector<Vertex>* vertices = solution->getVertices();
    
    outFile << "VECTORS dirichlet_border float" << endl;
    for (unsigned int i = 0; i < numberOfVertices; i++) {
        unsigned int index = filterNullVoxels ? vertexFilteredToOrigIndex[i] : i;
        DirichletBoundary boundary = problem->getDirichletBoundaryAtVertex(index);
        outFile << (boundary.isXFixed() ? 1 : 0) << " " << (boundary.isYFixed() ? 1 : 0) << " " << (boundary.isZFixed() ? 1 : 0) << endl;
    }
    outFile << endl;

    outFile << "VECTORS neumann_border float" << endl;
    for (unsigned int i = 0; i < numberOfVertices; i++) {
        unsigned int index = filterNullVoxels ? vertexFilteredToOrigIndex[i] : i;
        NeumannBoundary boundary = problem->getNeumannBoundaryAtVertex(index);
        outFile << boundary.stress.x << " " << boundary.stress.y << " " << boundary.stress.z << endl;
    }
    outFile << endl;
}

void VTKSolutionVisualizer::writeResiduals() {
    outFile << "VECTORS residual double" << endl;

    std::vector<Vertex>* vertices = solution->getVertices();
    for (unsigned int i = 0; i < numberOfVertices; i++) {
        unsigned int index = filterNullVoxels ? vertexFilteredToOrigIndex[i] : i;
        Vertex v = vertices->at(index);
        if (filterNullVoxels && v.materialConfigId == 0) {
            continue;
        }
        VertexCoordinate fullresCoord = solution->mapToCoordinate(index);
        REAL residual = impVol->getResidualOnLevel(0, (fullresCoord ) / 2);
        outFile << residual << " " << residual << " " << residual << endl;
    }
    outFile << endl;
}

void VTKSolutionVisualizer::writeMaterialConfigIds() {
    outFile << "SCALARS matconfigid float 1" << endl;
    outFile << "LOOKUP_TABLE default" << endl;

    std::vector<Vertex>* vertices = solution->getVertices();
    for (unsigned int i = 0; i < numberOfVertices; i++) {
        unsigned int index = filterNullVoxels ? vertexFilteredToOrigIndex[i] : i;
        Vertex v = vertices->at(index);
        if (filterNullVoxels && v.materialConfigId == 0) {
            continue;
        }
        VertexCoordinate fullresCoord = solution->mapToCoordinate(index);
        int configid = v.materialConfigId;
        outFile << configid << endl;
    }
    outFile << endl;
}
