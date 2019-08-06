#include "stdafx.h"

#include <cstdio>
#include <experimental/filesystem>
#include "XMLProblemDeserializer.h"
#include "problem/boundaryconditions/BoundaryProjector.h"
#include "io/ImageDirectoryImporter.h"

XMLProblemDeserializer::XMLProblemDeserializer(const std::string& path) : 
    document(),
    pathToFile(path)
{

}

std::string XMLProblemDeserializer::getFullPathToInputFile(std::string& relativePath) {
    namespace fs = std::experimental::filesystem;
    fs::path xmlPath(pathToFile);
    auto baseDir = fs::path();

    if (xmlPath.has_parent_path()) {
        baseDir = xmlPath.remove_filename();
    }
    
    do 
    {
        auto combinePath = baseDir / relativePath;
        if (fs::exists(combinePath))
        {
            return combinePath.string();
        }
        baseDir = baseDir.parent_path();
    } while (baseDir.has_parent_path());

    std::stringstream ss;
    ss << "Could not find the input file '" << relativePath << "' specified in '" << pathToFile << "'. Expected the input file to be in the same directory as the XML file with no leading / \n";

    throw std::ios_base::failure(ss.str());
}

std::unique_ptr<ProblemInstance> XMLProblemDeserializer::getProblemInstance() {
    std::unique_ptr<ProblemInstance> problemInstance = std::unique_ptr<ProblemInstance>(new ProblemInstance());

    try {
        
        document.LoadFile(pathToFile.c_str());

        parseMaterialDictionary(problemInstance);
        parseDiscreteProblem(problemInstance);
        parseDirichletBoundaryProjection(problemInstance);
        parseNeumannBoundaryProjection(problemInstance);
        parseLevelsOfDetail(problemInstance);
    }
    catch (std::exception e) {
        std::cerr << "[ERROR] XMLProblemDeserializer: " << e.what();
        throw std::runtime_error("Fatal error while deserializing XML file");
    }

    return problemInstance;
}

void XMLProblemDeserializer::parseMaterialDictionary(std::unique_ptr<ProblemInstance>& problemInstance) {
    tinyxml2::XMLElement* matDictElement = document.RootElement()->FirstChildElement("MaterialDictionary");
    if (matDictElement == NULL) {
        throw std::ios_base::failure("Required element MaterialDictionary not found");
    }

    MaterialDictionary dict;

    for (tinyxml2::XMLElement* child = matDictElement->FirstChildElement("Material"); child != NULL; child = child->NextSiblingElement("Material")) {
        unsigned int id = child->IntAttribute("id");
        if (id == 0) {
            throw std::ios_base::failure("Material ID 0 is reserved for empty voxels. Please use another ID");
        }
        REAL youngsModulus = child->FloatAttribute("youngsModulus");
        REAL poisson = child->FloatAttribute("poissonRatio");
        Material mat(youngsModulus, poisson, id);
        dict.addMaterial(mat);
    }

    problemInstance->materialDictionary = dict;
}

void XMLProblemDeserializer::parseMaterialMapping(std::unique_ptr<ProblemInstance>& problemInstance, MRCImporter* importer, tinyxml2::XMLElement* inputFileElement) {
    tinyxml2::XMLElement* matMappingElement = inputFileElement->FirstChildElement("MaterialMapping");
    if (matMappingElement == NULL) {
        throw std::ios_base::failure("A MaterialMapping is required when importing a problem from an MRC stack");
    }

    for (tinyxml2::XMLElement* child = matMappingElement->FirstChildElement("MaterialMap"); child != NULL; child = child->NextSiblingElement("MaterialMap")) {
        unsigned int matID = child->IntAttribute("materialID", 300);
        unsigned int colorValue = child->IntAttribute("colorValue", 300);

        if (matID > 255) {
            throw std::ios_base::failure("Invalid material ID in material mapping. Valid IDs must be in range (0,255)");
        }
        if (colorValue > 255) {
            throw std::ios_base::failure("Invalid color value in material mapping. Valid color values must be in range (0,255)");
        }

        Material* mat = problemInstance->materialDictionary.getMaterialById(matID);
        if (mat == NULL) {
            throw std::ios_base::failure("Encountered a material ID that has no matching material in the given MaterialDictionary");
        }
        
        importer->addMaterialMapping(mat, (unsigned char)colorValue);
    }

}

void XMLProblemDeserializer::parseDiscreteProblem(std::unique_ptr<ProblemInstance>& problemInstance) {
    tinyxml2::XMLElement* problemElement = document.RootElement()->FirstChildElement("DiscreteProblem");
    if (problemElement == NULL) {
        throw std::ios_base::failure("Required element DiscreteProblem not found");
    }

    unsigned int sizeX = problemElement->IntAttribute("sizeX", 0);
    unsigned int sizeY = problemElement->IntAttribute("sizeY", 0);
    unsigned int sizeZ = problemElement->IntAttribute("sizeZ", 0);

    REAL voxelSizeX = problemElement->FloatAttribute("voxelSizeX", 0);
    REAL voxelSizeY = problemElement->FloatAttribute("voxelSizeY", 0);
    REAL voxelSizeZ = problemElement->FloatAttribute("voxelSizeZ", 0);
    
    if (sizeX == 0 || sizeY == 0 || sizeZ == 0) {
        throw std::ios_base::failure("Invalid or missing problem size encountered while deserializing DiscreteProblem");
    }
    if (voxelSizeX == 0 || voxelSizeY == 0 || voxelSizeZ == 0) {
        throw std::ios_base::failure("Invalid or missing voxel size encountered while deserializing DiscreteProblem");
    }

    problemInstance->initFromParameters(libmmv::Vec3ui(sizeX, sizeY, sizeZ), libmmv::Vec3<REAL>(voxelSizeX, voxelSizeY, voxelSizeZ));

    parseInputFile(problemInstance, problemElement);
}

void XMLProblemDeserializer::parseInputFile(std::unique_ptr<ProblemInstance>& problemInstance, tinyxml2::XMLElement* discreteProblemElement) {
    tinyxml2::XMLElement* inputFileElement = discreteProblemElement->FirstChildElement("InputFile");
    if (inputFileElement == NULL) {
        return;
    }

    const char* inputFilePath = inputFileElement->Attribute("inputFileName");
    std::string filePathString(inputFilePath);
    filePathString = getFullPathToInputFile(filePathString);

    if (filePathString.find(".mrc") != std::string::npos || filePathString.find(".MRC") != std::string::npos) {
        MRCImporter importer(filePathString);
        parseMaterialMapping(problemInstance, &importer, inputFileElement);
        importer.populateDiscreteProblem(problemInstance->getProblemLOD(0));
    }
    else {

    }

}

void XMLProblemDeserializer::parseDirichletBoundaryProjection(std::unique_ptr<ProblemInstance>& problemInstance) {
    tinyxml2::XMLElement* boundariesElement = document.RootElement()->FirstChildElement("DirichletBoundaries");
    if (boundariesElement == NULL) {
        throw std::ios_base::failure("Required element DirichletBoundaries not found");
    }

    BoundaryProjector bProjector(problemInstance->getProblemLOD(0), ProblemSide::NEGATIVE_Z);
    tinyxml2::XMLElement* projectorElement = boundariesElement->FirstChildElement("DirichletBoundaryProjector");
    if (projectorElement != NULL) {
        int maxDepth = projectorElement->IntAttribute("maximumDepth", 10);
        bProjector.setMaxProjectionDepth(maxDepth);
    }
    else {
        std::cout << "WARN: No DirichletBoundaryProjector element found, using defaults maxDepth=10 and projectionDirection=-Z \n";
    }

    for (tinyxml2::XMLElement* child = boundariesElement->FirstChildElement("DirichletBoundary"); child != NULL; child = child->NextSiblingElement("DirichletBoundary")) {
        DirichletBoundary::Condition fixed = DirichletBoundary::NONE;

        if (child->BoolAttribute("fixedX", false)) {
            fixed = (DirichletBoundary::Condition)(fixed | DirichletBoundary::FIXED_X);
        } 
        if (child->BoolAttribute("fixedY", false)) {
            fixed = (DirichletBoundary::Condition)(fixed | DirichletBoundary::FIXED_Y);
        }
        if (child->BoolAttribute("fixedZ", false)) {
            fixed = (DirichletBoundary::Condition)(fixed | DirichletBoundary::FIXED_Z);
        }

        DirichletBoundary boundary(fixed);

        const char* direction = child->Attribute("projectionDirection");
        if (direction == NULL) {
            throw std::ios_base::failure("Invalid or missing projectionDirection attribute in DirichletBoundary");
        }

        std::string directionVal(direction);

        if (directionVal == "+x" || directionVal == "+X") {
            bProjector.setProjectionDirection(ProblemSide::POSITIVE_X);
        }
        else if (directionVal == "-x" || directionVal == "-X") {
            bProjector.setProjectionDirection(ProblemSide::NEGATIVE_X);
        }
        else if (directionVal == "+y" || directionVal == "+Y") {
            bProjector.setProjectionDirection(ProblemSide::POSITIVE_Y);
        }
        else if (directionVal == "-y" || directionVal == "-Y") {
            bProjector.setProjectionDirection(ProblemSide::NEGATIVE_Y);
        }
        else if (directionVal == "+z" || directionVal == "+Z") {
            bProjector.setProjectionDirection(ProblemSide::POSITIVE_Z);
        }
        else if (directionVal == "-z" || directionVal == "-Z") {
            bProjector.setProjectionDirection(ProblemSide::NEGATIVE_Z);
        }

        bProjector.projectDirichletBoundary(&boundary);
    }

}

void XMLProblemDeserializer::parseNeumannBoundaryProjection(std::unique_ptr<ProblemInstance>& problemInstance) {
    tinyxml2::XMLElement* boundariesElement = document.RootElement()->FirstChildElement("NeumannBoundaries");
    if (boundariesElement == NULL) {
        throw std::ios_base::failure("Required element NeumannBoundaries not found");
    }

    BoundaryProjector bProjector(problemInstance->getProblemLOD(0), ProblemSide::NEGATIVE_Z);
    unsigned char materialFilter = 255;
    tinyxml2::XMLElement* projectorElement = boundariesElement->FirstChildElement("NeumannBoundaryProjector");
    if (projectorElement != NULL) {
        int maxDepth = projectorElement->IntAttribute("maximumDepth", 10);
        materialFilter = (unsigned char) projectorElement->IntAttribute("materialFilter", 255);
        bProjector.setMaxProjectionDepth(maxDepth);
    }
    else {
        std::cout << "[WARN]: No NeumannBoundaryProjector element found, using defaults maxDepth=10 and projectionDirection=-Z \n";
    }

    for (tinyxml2::XMLElement* child = boundariesElement->FirstChildElement("NeumannBoundary"); child != NULL; child = child->NextSiblingElement("NeumannBoundary")) {
        
        REAL totalStressInNewtons = child->FloatAttribute("totalForceInNewtons", asREAL(0.0));
        if (totalStressInNewtons == 0) {
            throw std::ios_base::failure("invalid or missing totalForceInNewtons attribute in NeumannBoundary");
        }

        const char* direction = child->Attribute("projectionDirection");
        if (direction == NULL) {
            throw std::ios_base::failure("invalid or missing projectionDirection attribute in NeumannBoundary");
        }

        std::string directionVal(direction);

        if (directionVal == "+x" || directionVal == "+X") {
            bProjector.setProjectionDirection(ProblemSide::POSITIVE_X);
        }
        else if (directionVal == "-x" || directionVal == "-X") {
            bProjector.setProjectionDirection(ProblemSide::NEGATIVE_X);
        }
        else if (directionVal == "+y" || directionVal == "+Y") {
            bProjector.setProjectionDirection(ProblemSide::POSITIVE_Y);
        }
        else if (directionVal == "-y" || directionVal == "-Y") {
            bProjector.setProjectionDirection(ProblemSide::NEGATIVE_Y);
        }
        else if (directionVal == "+z" || directionVal == "+Z") {
            bProjector.setProjectionDirection(ProblemSide::POSITIVE_Z);
        }
        else if (directionVal == "-z" || directionVal == "-Z") {
            bProjector.setProjectionDirection(ProblemSide::NEGATIVE_Z);
        }

        bProjector.projectNeumannBoundary(totalStressInNewtons, materialFilter);
    }

}

void XMLProblemDeserializer::parseLevelsOfDetail(std::unique_ptr<ProblemInstance>& problemInstance) {
    tinyxml2::XMLElement* lodGenElement = document.RootElement()->FirstChildElement("LODGenerator");
    if (lodGenElement == NULL) {
        throw std::ios_base::failure("Required element LODGenerator not found");
    }

    int levelsOfDetail = lodGenElement->IntAttribute("numLevelsOfDetail", 0);
    problemInstance->createAdditionalLODs(levelsOfDetail);
}
