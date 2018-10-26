#pragma once

#include <iostream>

#include "problem/ProblemInstance.h"
#include "problem/DiscreteProblem.h"
#include "material/MaterialDictionary.h"
#include "io/tinyxml2/tinyxml2.h"
#include "io/MRCImporter.h"

class XMLProblemDeserializer {
public:
    XMLProblemDeserializer(const std::string& path);

    ProblemInstance getProblemInstance();
    std::string getFullPathToInputFile(std::string& relativePath);

protected:
    std::string pathToFile;
    tinyxml2::XMLDocument document;
    ProblemInstance problemInstance;
    MRCImporter mrcImporter;

    void parseMaterialDictionary();
    void parseMaterialMapping(MRCImporter* importer, tinyxml2::XMLElement* discreteProblemElement);
    void parseDiscreteProblem();
    void parseDirichletBoundaryProjection();
    void parseNeumannBoundaryProjection();
    void parseLevelsOfDetail();

};
