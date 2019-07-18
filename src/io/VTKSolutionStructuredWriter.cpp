#include "stdafx.h"

#include "VTKSolutionWriter.h"
#include "VTKSolutionStructuredWriter.h"

VTKSolutionStructuredWriter::~VTKSolutionStructuredWriter()
{
}

void VTKSolutionStructuredWriter::writeEntireStructureToFile(const std::string& filename)
{
	if (nullVoxelsWereFiltered) {
		throw std::runtime_error("Null voxels must not be filtered for structured grid format");
	}

	VTKSolutionWriter::writeEntireStructureToFile(filename);
}

void VTKSolutionStructuredWriter::writeEntireStructureToStream(std::ostream& stream)
{
	writeHeader(stream);
	writePoints(stream);
	writeCellData(stream);
}

void VTKSolutionStructuredWriter::writeHeader(std::ostream& stream)
{
	const DiscreteProblem* problem = solution->getProblem();
	stream << "# vtk DataFile Version 2.0" << std::endl;
	stream << "Stochastic Mechanical Solver Debug Output" << std::endl;
	stream << "ASCII" << std::endl;
	stream << "DATASET STRUCTURED_GRID" << std::endl;

	//VTK format for structured grids requires an offset of 1
	stream << "DIMENSIONS" << " " << problem->getSize().x + 1 << " " << problem->getSize().y + 1 << " " << problem->getSize().z + 1 << std::endl << std::endl;

}
