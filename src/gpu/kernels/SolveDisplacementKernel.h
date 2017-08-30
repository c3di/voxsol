#pragma once
#include <stdafx.h>
#include <vector>
#include <assert.h>
#include <memory>

#include "CudaKernel.h"
#include "libmmv/math/Vec3.h"
#include "solution/Solution.h"

extern "C" void cudaLaunchSolveDisplacementKernel(Vertex* verticesOnGPU, REAL* matConfigEquationsOnGPU, unsigned int numVertices);

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

    Vertex* serializedVertices;
    REAL* serializedMatConfigEquations;
    unsigned short* constantSolutionDimensions;

    void prepareInputs();

    void pushMatConfigEquations();
    void pushVertices();
    void pushSolutionDimensions();

    void pullVertices();

    void serializeMaterialConfigurationEquations(void* destination);

};
