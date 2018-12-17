#pragma once
#include "CudaKernel.h"
#include "gpu/sampling/ResidualVolume.h"
#include "solution/Solution.h"

extern "C" void cudaLaunchFullResidualUpdateKernel(Vertex* verticesOnGPU, REAL* residualsLevelZeroOnGPU, REAL* matConfigEquationsOnGPU, const uint3 solutionDims);

class FullResidualUpdateKernel : public CudaKernel {
public:

    FullResidualUpdateKernel(Solution* sol, ResidualVolume* resVol, Vertex* verticesOnGPU, REAL* matConfigEquationsOnGPU);
    ~FullResidualUpdateKernel();

    void launch() override;
    void setMatConfigEquationsOnGPU(REAL* equationsOnGPU);
    void setVerticesOnGPU(Vertex* verticesOnGPU);

protected:
    Solution* solution;
    ResidualVolume* residualVolume;
    Vertex* verticesOnGPU;
    REAL* matConfigEquationsOnGPU;
    uint3 solutionDimensions;
    uint3 residualLevelZeroDimensions;


    bool canExecute() override;
    void freeCudaResources() override;
};

