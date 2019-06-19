#pragma once

#include "VTKSolutionWriter.h"

class VTKSolutionStructuredWriter :
	public VTKSolutionWriter
{
public:
	VTKSolutionStructuredWriter(Solution* solution, ResidualVolume* importanceVolume = nullptr) : VTKSolutionWriter(solution, importanceVolume) {};
	~VTKSolutionStructuredWriter();

	void VTKSolutionStructuredWriter::writeEntireStructureToFile(const std::string& filename);

protected:
	void VTKSolutionStructuredWriter::writeEntireStructureToStream(std::ostream& stream);
	void VTKSolutionStructuredWriter::writeHeader(std::ostream& stream);
};


