#include "stdafx.h"

#include "CSVSolutionWriter.h"
#include "solution/SolutionAnalyzer.h"

CSVSolutionWriter::CSVSolutionWriter(Solution * solution) :
    solution(solution)
{
}

CSVSolutionWriter::~CSVSolutionWriter() {

}

void CSVSolutionWriter::writeSolutionToFile(const std::string & filepath) {
    std::ofstream stream(filepath, std::ios::out);
    std::cout << "Writing CSV solution file to " << filepath << " ...";

    try
    {
        writeSolutionToStream(stream);
    }
    catch (std::ifstream::failure exception)
    {
        std::cerr << exception.what() << "\n";
        stream.close();
        throw exception;
    }
    stream.close();
    
    std::cout << "Done.\n";
}

void CSVSolutionWriter::writeSolutionToStream(std::ostream & stream) {
    SolutionAnalyzer solutionAnalyzer(solution->getProblem(), solution);
    DiscreteProblem* problem = solution->getProblem();

    stream << "voxel_index,material_id,von_Mises_strain,strain_e_xx,strain_e_yy,strain_e_zz,strain_gamma_yz,strain_gamma_xz,strain_gamma_xy,von_Mises_stress,stress_sigma_xx,stress_sigma_yy,stress_sigma_zz,stress_tau_yz,stress_tau_xz,stress_tau_xy\n";

    int voxelIndex = 0;
    
    for (unsigned int z = 0; z < problem->getSize().z; z++) {
        for (unsigned int y = 0; y < problem->getSize().y; y++) {
            for (unsigned int x = 0; x < problem->getSize().x; x++) {
                VoxelCoordinate coord(x, y, z);

                stream << voxelIndex << "," << static_cast<unsigned int>(problem->getMaterial(coord)->id) << ",";

                writeStrainTensorsForVoxel(stream, coord, solutionAnalyzer);
                stream << ",";
                writeStressTensorsForVoxel(stream, coord, solutionAnalyzer);

                stream << std::endl;

                voxelIndex++;
            }
        }
    }
}

void CSVSolutionWriter::writeStrainTensorsForVoxel(std::ostream & stream, const VoxelCoordinate & coord, SolutionAnalyzer& solutionAnalyzer) {
    stream << solutionAnalyzer.getVonMisesStrainAt(coord) << ",";
   
    REAL* strainTensor = solutionAnalyzer.getStrainTensorAt(coord);   
    stream << strainTensor[0] << "," << strainTensor[1] << "," << strainTensor[2] << "," << strainTensor[3] << "," << strainTensor[4] << "," << strainTensor[5];
}

void CSVSolutionWriter::writeStressTensorsForVoxel(std::ostream & stream, const VoxelCoordinate & coord, SolutionAnalyzer& solutionAnalyzer) {
    stream << solutionAnalyzer.getVonMisesStressAt(coord) << ",";

    REAL* stressTensor = solutionAnalyzer.getStressTensorAt(coord);
    stream << stressTensor[0] << "," << stressTensor[1] << "," << stressTensor[2] << "," << stressTensor[3] << "," << stressTensor[4] << "," << stressTensor[5];
}
