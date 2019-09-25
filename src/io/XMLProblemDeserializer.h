#pragma once

#include <iostream>

#include "problem/ProblemInstance.h"
#include "problem/DiscreteProblem.h"
#include "material/MaterialDictionary.h"
#include "io/tinyxml2/tinyxml2.h"
#include "io/MRCImporter.h"

class DisplacementBoundary;
enum ProblemSide;

class XMLProblemDeserializer {
public:
    XMLProblemDeserializer(const std::string& path);

    std::unique_ptr<ProblemInstance> getProblemInstance();
    std::string getFullPathToInputFile(std::string& relativePath);

protected:
    std::string pathToFile;
    tinyxml2::XMLDocument document;
    MRCImporter mrcImporter;

    void parseMaterialDictionary(std::unique_ptr<ProblemInstance>& problemInstance);
    void parseMaterialMapping(std::unique_ptr<ProblemInstance>& problemInstance, MRCImporter* importer, tinyxml2::XMLElement* inputFileElement);
    void parseDiscreteProblem(std::unique_ptr<ProblemInstance>& problemInstance);
    void parseDirichletBoundaryProjection(std::unique_ptr<ProblemInstance>& problemInstance);
    void parseNeumannBoundaryProjection(std::unique_ptr<ProblemInstance>& problemInstance);
    void parseDisplacementBoundaryProjection(std::unique_ptr<ProblemInstance>& problemInstance);
    void parseLevelsOfDetail(std::unique_ptr<ProblemInstance>& problemInstance);
    void parseInputFile(std::unique_ptr<ProblemInstance>& problemInstance, tinyxml2::XMLElement* discreteProblemElement);
    DisplacementBoundary getDisplacementBoundaryFromPercent(std::unique_ptr<ProblemInstance>& problemInstance, REAL percentOfDimension, ProblemSide& projectFrom);
};
