#pragma once
#include "VTKSolutionWriter.h"
class VTKSolutionStructuredWriter :
	public VTKSolutionWriter
{
public:
	VTKSolutionStructuredWriter() : VTKSolutionWriter(solution, importanceVolume) {};
	~VTKSolutionStructuredWriter();
};

