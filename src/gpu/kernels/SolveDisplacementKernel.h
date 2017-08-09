#pragma once
#include <stdafx.h>
#include <vector>
#include <assert.h>
#include <memory>

#include "CudaKernel.h"
#include "libmmv/math/Vec3.h"
#include "solution/Solution.h"

extern "C" void cudaLaunchSolveDisplacementKernel(REAL* displacementsOnGPU, ConfigId* matConfigEquationIdsOnGPU, REAL* matConfigEquationsOnGPU, unsigned int numVertices);

class SolveDisplacementKernel : public CudaKernel {

public:

    SolveDisplacementKernel(Solution* sol);
    ~SolveDisplacementKernel();

    void launch() override;

protected:

    bool canExecute() override;
    void freeCudaResources();

private:
    Solution* solution;

    ConfigId* matConfigEquationIdsOnGPU;
    REAL* displacementsOnGPU;
    REAL* matConfigEquationsOnGPU;

    void prepareInputs();

    void pushMatConfigEquationIds();
    void pushDisplacements();
    void pushMatConfigEquations();

    void pullDisplacements();

    void serializeMaterialConfigurationEquations(void* destination);

};
