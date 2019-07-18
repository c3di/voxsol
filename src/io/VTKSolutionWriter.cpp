#include "stdafx.h"

#include "VTKSolutionWriter.h"

#include <iomanip>
#include <iostream>
#include <fstream>

#include "solution/SolutionAnalyzer.h"

VTKSolutionWriter::VTKSolutionWriter(Solution* solution, ResidualVolume* importanceVolume) :
	solution(solution),
	numberOfCells(solution->getProblem()->getNumberOfVoxels()),
	numberOfPoints(static_cast<unsigned int>(solution->getVertices()->size())),
	importanceVolume(importanceVolume)
{
}

VTKSolutionWriter::~VTKSolutionWriter()
{
}

void VTKSolutionWriter::filterOutNullVoxels()
{
	fillFilteredPointMap();
	fillFilteredCellMap();
	nullVoxelsWereFiltered = true;
}

void VTKSolutionWriter::setMechanicalValuesOutput(bool flag)
{
	enableMechanicalValuesOutput = flag;
}

void VTKSolutionWriter::fillFilteredPointMap()
{
	numberOfPoints = 0;
	auto vertices = solution->getVertices();
	for (unsigned int i = 0; i < vertices->size(); i++)
	{
		if (vertices->at(i).materialConfigId == 0)
			continue;

		pointMapOriginalToFiltered[i] = numberOfPoints;
		pointMapFilteredToOriginal[numberOfPoints] = i;
		numberOfPoints++;
	}
}

void VTKSolutionWriter::fillFilteredCellMap()
{
	numberOfCells = 0;
	DiscreteProblem* problem = solution->getProblem();
	for (unsigned int i = 0; i < problem->getNumberOfVoxels(); i++)
	{
		if (problem->getMaterial(i)->id == Material::EMPTY.id)
			continue;

		cellMapOriginalToFiltered[i] = numberOfCells;
		cellMapFilteredToOriginal[numberOfCells] = i;
		numberOfCells++;
	}
}

unsigned int VTKSolutionWriter::getMappedIndex(unsigned int originalIndex)
{
	if (!nullVoxelsWereFiltered)
		return originalIndex;

	return pointMapOriginalToFiltered[originalIndex];
}

void VTKSolutionWriter::writeEntireStructureToFile(const std::string& filename)
{
	std::ofstream stream(filename, std::ios::out);
	try
	{
		writeEntireStructureToStream(stream);
	}
	catch (std::ifstream::failure exception)
	{
		stream.close();
		throw exception;
	}
	stream.close();
}

void VTKSolutionWriter::writeEntireStructureToStream(std::ostream& stream)
{
	writeHeader(stream);
	writePoints(stream);
	writeCells(stream);
	writeCellTypes(stream);
	writeCellData(stream);
	writePointData(stream);
}

void VTKSolutionWriter::writeHeader(std::ostream& stream)
{
	stream << "# vtk DataFile Version 2.0" << std::endl;
	stream << "Stochastic Mechanical Solver Debug Output" << std::endl;
	stream << "ASCII" << std::endl;
	stream << "DATASET UNSTRUCTURED_GRID" << std::endl;
}

void VTKSolutionWriter::writePoints(std::ostream& stream)
{
	stream << "POINTS " << numberOfPoints << " double" << std::endl;
	for (unsigned int i = 0; i < numberOfPoints; i++)
	{
		writeOnePoint(stream, i);
	}
	stream << std::endl;
}

void VTKSolutionWriter::writeOnePoint(std::ostream& stream, unsigned int originalIndex)
{
	unsigned int mappedIndex = getMappedIndex(originalIndex);

	if (nullVoxelsWereFiltered && solution->getVertices()->at(mappedIndex).materialConfigId == 0)
		return;

	libmmv::Vec3<REAL> position = solution->getProblem()->getVertexPosition(mappedIndex);
	stream << position.x << " " << position.y << " " << position.z << " " << std::endl;
}

void VTKSolutionWriter::writeCells(std::ostream& stream) {
	const DiscreteProblem* problem = solution->getProblem();

	stream << "CELLS " << numberOfCells << " " << numberOfCells * 9 << std::endl;
	for (unsigned int zi = 0; zi < problem->getSize().z; zi++) {
		for (unsigned int yi = 0; yi < problem->getSize().y; yi++) {
			for (unsigned int xi = 0; xi < problem->getSize().x; xi++) {
				VoxelCoordinate coord(xi, yi, zi);
				unsigned int flatIndex = problem->mapToVoxelIndex(coord);
				if (nullVoxelsWereFiltered && cellMapOriginalToFiltered.count(flatIndex) <= 0) {
					// There is no mapping for this voxel index, so it must have been filtered out due to null material
					continue;
				}
				writeCell(stream, coord);
			}
		}
	}
	stream << std::endl;
}

void VTKSolutionWriter::writeCell(std::ostream& stream, VoxelCoordinate& coord) {
	stream << "8 ";
	writeVertexToCell(stream, 0, 0, 0, coord);
	writeVertexToCell(stream, 1, 0, 0, coord);
	writeVertexToCell(stream, 1, 1, 0, coord);
	writeVertexToCell(stream, 0, 1, 0, coord);
	writeVertexToCell(stream, 0, 0, 1, coord);
	writeVertexToCell(stream, 1, 0, 1, coord);
	writeVertexToCell(stream, 1, 1, 1, coord);
	writeVertexToCell(stream, 0, 1, 1, coord);
	stream << std::endl;
}

void VTKSolutionWriter::writeVertexToCell(std::ostream& stream, unsigned int xi, unsigned int yi, unsigned int zi, VoxelCoordinate& coord) {
	VertexCoordinate corner = coord + VoxelCoordinate(xi, yi, zi);
	int flatIndexOfCorner = solution->getProblem()->mapToVertexIndex(corner);
	if (nullVoxelsWereFiltered) {
		if (pointMapOriginalToFiltered.count(flatIndexOfCorner) <= 0) {
			throw std::exception("Could not find mapping for vertex");
		}
		flatIndexOfCorner = pointMapOriginalToFiltered[flatIndexOfCorner];
	}
	stream << flatIndexOfCorner << " ";
}

void VTKSolutionWriter::writeCellTypes(std::ostream& stream) {
	stream << "CELL_TYPES " << numberOfCells << std::endl;
	for (unsigned int i = 0; i < numberOfCells; i++) {
		stream << "12" << std::endl;
	}
	stream << std::endl;
}

void VTKSolutionWriter::writeCellData(std::ostream& stream) {
	stream << "CELL_DATA " << numberOfCells << std::endl;

	writeMaterials(stream);

	if (enableMechanicalValuesOutput) {
		std::cout << "Writing Mechanical values..." << std::endl;
		SolutionAnalyzer solutionAnalyzer(solution->getProblem(), solution);
		writeVonMisesStresses(stream, &solutionAnalyzer);
		writeVonMisesStrains(stream, &solutionAnalyzer);
		writeStressTensors(stream, &solutionAnalyzer);
		writeStrainTensors(stream, &solutionAnalyzer);
	}
}

void VTKSolutionWriter::writeMaterials(std::ostream& stream) {
	stream << "SCALARS material_id int 1" << std::endl;
	stream << "LOOKUP_TABLE default" << std::endl;

	DiscreteProblem* problem = solution->getProblem();
	for (unsigned int i = 0; i < numberOfCells; i++) {
		unsigned int index = nullVoxelsWereFiltered ? cellMapFilteredToOriginal[i] : i;
		int matId = static_cast<int>(problem->getMaterial(index)->id);
		stream << matId << std::endl;
	}
	stream << std::endl;
}

void VTKSolutionWriter::writeVonMisesStresses(std::ostream& stream, SolutionAnalyzer* solutionAnalyzer) {
	stream << "SCALARS von_Mises_stress float 1" << std::endl;
	stream << "LOOKUP_TABLE default" << std::endl;

	for (unsigned int i = 0; i < numberOfCells; i++) {
		unsigned int index = nullVoxelsWereFiltered ? cellMapFilteredToOriginal[i] : i;
		REAL stress = static_cast<REAL>(solutionAnalyzer->getVonMisesStressAt(index));
		stream << stress << std::endl;
	}
	stream << std::endl;
}

void VTKSolutionWriter::writeVonMisesStrains(std::ostream& stream, SolutionAnalyzer* solutionAnalyzer) {
	stream << "SCALARS von_Mises_strain float 1" << std::endl;
	stream << "LOOKUP_TABLE default" << std::endl;

	for (unsigned int i = 0; i < numberOfCells; i++) {
		unsigned int index = nullVoxelsWereFiltered ? cellMapFilteredToOriginal[i] : i;
		REAL stress = static_cast<REAL>(solutionAnalyzer->getVonMisesStrainAt(index));
		stream << stress << std::endl;
	}
	stream << std::endl;
}

void VTKSolutionWriter::writeStressTensors(std::ostream& stream, SolutionAnalyzer* solutionAnalyzer) {
	stream << "SCALARS stress_sigma_xx float 1" << std::endl;
	stream << "LOOKUP_TABLE default" << std::endl;

	for (unsigned int i = 0; i < numberOfCells; i++) {
		unsigned int index = nullVoxelsWereFiltered ? cellMapFilteredToOriginal[i] : i;
		REAL* stressTensor = solutionAnalyzer->getStressTensorAt(index);
		stream << stressTensor[0] << std::endl;
	}
	stream << std::endl;

	stream << "SCALARS stress_sigma_yy float 1" << std::endl;
	stream << "LOOKUP_TABLE default" << std::endl;

	for (unsigned int i = 0; i < numberOfCells; i++) {
		unsigned int index = nullVoxelsWereFiltered ? cellMapFilteredToOriginal[i] : i;
		REAL* stressTensor = solutionAnalyzer->getStressTensorAt(index);
		stream << stressTensor[1] << std::endl;
	}
	stream << std::endl;

	stream << "SCALARS stress_sigma_zz float 1" << std::endl;
	stream << "LOOKUP_TABLE default" << std::endl;

	for (unsigned int i = 0; i < numberOfCells; i++) {
		unsigned int index = nullVoxelsWereFiltered ? cellMapFilteredToOriginal[i] : i;
		REAL* stressTensor = solutionAnalyzer->getStressTensorAt(index);
		stream << stressTensor[2] << std::endl;
	}
	stream << std::endl;

	stream << "SCALARS stress_tau_yz float 1" << std::endl;
	stream << "LOOKUP_TABLE default" << std::endl;

	for (unsigned int i = 0; i < numberOfCells; i++) {
		unsigned int index = nullVoxelsWereFiltered ? cellMapFilteredToOriginal[i] : i;
		REAL* stressTensor = solutionAnalyzer->getStressTensorAt(index);
		stream << stressTensor[3] << std::endl;
	}
	stream << std::endl;

	stream << "SCALARS stress_tau_xz float 1" << std::endl;
	stream << "LOOKUP_TABLE default" << std::endl;

	for (unsigned int i = 0; i < numberOfCells; i++) {
		unsigned int index = nullVoxelsWereFiltered ? cellMapFilteredToOriginal[i] : i;
		REAL* stressTensor = solutionAnalyzer->getStressTensorAt(index);
		stream << stressTensor[4] << std::endl;
	}
	stream << std::endl;

	stream << "SCALARS stress_tau_xy float 1" << std::endl;
	stream << "LOOKUP_TABLE default" << std::endl;

	for (unsigned int i = 0; i < numberOfCells; i++) {
		unsigned int index = nullVoxelsWereFiltered ? cellMapFilteredToOriginal[i] : i;
		REAL* stressTensor = solutionAnalyzer->getStressTensorAt(index);
		stream << stressTensor[5] << std::endl;
	}
	stream << std::endl;
}

void VTKSolutionWriter::writeStrainTensors(std::ostream& stream, SolutionAnalyzer* solutionAnalyzer) {
	stream << "SCALARS strain_e_xx float 1" << std::endl;
	stream << "LOOKUP_TABLE default" << std::endl;

	for (unsigned int i = 0; i < numberOfCells; i++) {
		unsigned int index = nullVoxelsWereFiltered ? cellMapFilteredToOriginal[i] : i;
		REAL* strainTensor = solutionAnalyzer->getStrainTensorAt(index);
		stream << strainTensor[0] << std::endl;
	}
	stream << std::endl;

	stream << "SCALARS strain_e_yy float 1" << std::endl;
	stream << "LOOKUP_TABLE default" << std::endl;

	for (unsigned int i = 0; i < numberOfCells; i++) {
		unsigned int index = nullVoxelsWereFiltered ? cellMapFilteredToOriginal[i] : i;
		REAL* strainTensor = solutionAnalyzer->getStrainTensorAt(index);
		stream << strainTensor[1] << std::endl;
	}
	stream << std::endl;

	stream << "SCALARS strain_e_zz float 1" << std::endl;
	stream << "LOOKUP_TABLE default" << std::endl;

	for (unsigned int i = 0; i < numberOfCells; i++) {
		unsigned int index = nullVoxelsWereFiltered ? cellMapFilteredToOriginal[i] : i;
		REAL* strainTensor = solutionAnalyzer->getStrainTensorAt(index);
		stream << strainTensor[2] << std::endl;
	}
	stream << std::endl;

	stream << "SCALARS strain_gamma_yz float 1" << std::endl;
	stream << "LOOKUP_TABLE default" << std::endl;

	for (unsigned int i = 0; i < numberOfCells; i++) {
		unsigned int index = nullVoxelsWereFiltered ? cellMapFilteredToOriginal[i] : i;
		REAL* strainTensor = solutionAnalyzer->getStrainTensorAt(index);
		stream << strainTensor[3] << std::endl;
	}
	stream << std::endl;

	stream << "SCALARS strain_gamma_xz float 1" << std::endl;
	stream << "LOOKUP_TABLE default" << std::endl;

	for (unsigned int i = 0; i < numberOfCells; i++) {
		unsigned int index = nullVoxelsWereFiltered ? cellMapFilteredToOriginal[i] : i;
		REAL* strainTensor = solutionAnalyzer->getStrainTensorAt(index);
		stream << strainTensor[4] << std::endl;
	}
	stream << std::endl;

	stream << "SCALARS strain_gamma_xy float 1" << std::endl;
	stream << "LOOKUP_TABLE default" << std::endl;

	for (unsigned int i = 0; i < numberOfCells; i++) {
		unsigned int index = nullVoxelsWereFiltered ? cellMapFilteredToOriginal[i] : i;
		REAL* strainTensor = solutionAnalyzer->getStrainTensorAt(index);
		stream << strainTensor[5] << std::endl;
	}
	stream << std::endl;
}

void VTKSolutionWriter::writePointData(std::ostream& stream) {
	stream << "POINT_DATA " << numberOfPoints << std::endl;

	writeDisplacements(stream);
	//writeBoundaries();

	if (enableResidualOutput && importanceVolume != nullptr) {
		writeResiduals(stream);
	}
	if (enableMatConfigIdOutput) {
		writeMaterialConfigIds(stream);
	}
}

void VTKSolutionWriter::writeDisplacements(std::ostream& stream) {
	stream << "VECTORS displacement double" << std::endl;

	std::vector<Vertex>* vertices = solution->getVertices();
	for (unsigned int i = 0; i < numberOfPoints; i++) {
		unsigned int index = nullVoxelsWereFiltered ? pointMapFilteredToOriginal[i] : i;
		Vertex v = vertices->at(index);
		stream << v.x << " " << v.y << " " << v.z << std::endl;
	}
	stream << std::endl;
}

void VTKSolutionWriter::writeBoundaries(std::ostream& stream) {
	DiscreteProblem* problem = solution->getProblem();
	std::vector<Vertex>* vertices = solution->getVertices();

	stream << "VECTORS dirichlet_border float" << std::endl;
	for (unsigned int i = 0; i < numberOfPoints; i++) {
		unsigned int index = nullVoxelsWereFiltered ? pointMapFilteredToOriginal[i] : i;
		DirichletBoundary boundary = problem->getDirichletBoundaryAtVertex(index);
		stream << (boundary.isXFixed() ? 1 : 0) << " " << (boundary.isYFixed() ? 1 : 0) << " " << (boundary.isZFixed() ? 1 : 0) << std::endl;
	}
	stream << std::endl;

	stream << "VECTORS neumann_border float" << std::endl;
	for (unsigned int i = 0; i < numberOfPoints; i++) {
		unsigned int index = nullVoxelsWereFiltered ? pointMapFilteredToOriginal[i] : i;
		NeumannBoundary boundary = problem->getNeumannBoundaryAtVertex(index);
		stream << boundary.stress.x << " " << boundary.stress.y << " " << boundary.stress.z << std::endl;
	}
	stream << std::endl;
}

void VTKSolutionWriter::writeResiduals(std::ostream& stream) {
	stream << "VECTORS residual double" << std::endl;

	std::vector<Vertex>* vertices = solution->getVertices();
	for (unsigned int i = 0; i < numberOfPoints; i++) {
		unsigned int index = nullVoxelsWereFiltered ? pointMapFilteredToOriginal[i] : i;
		Vertex v = vertices->at(index);
		if (nullVoxelsWereFiltered && v.materialConfigId == 0) {
			continue;
		}
		VertexCoordinate fullresCoord = solution->mapToCoordinate(index);
		REAL residual = importanceVolume->getResidualOnLevel(0, (fullresCoord) / 2);
		stream << residual << " " << residual << " " << residual << std::endl;
	}
	stream << std::endl;
}

void VTKSolutionWriter::writeMaterialConfigIds(std::ostream& stream) {
	stream << "SCALARS matconfigid int 1" << std::endl;
	stream << "LOOKUP_TABLE default" << std::endl;

	std::vector<Vertex>* vertices = solution->getVertices();
	for (unsigned int i = 0; i < numberOfPoints; i++) {
		unsigned int index = nullVoxelsWereFiltered ? pointMapFilteredToOriginal[i] : i;
		Vertex v = vertices->at(index);
		if (nullVoxelsWereFiltered && v.materialConfigId == 0) {
			continue;
		}
		VertexCoordinate fullresCoord = solution->mapToCoordinate(index);
		int configid = v.materialConfigId;
		stream << configid << std::endl;
	}
	stream << std::endl;
}
