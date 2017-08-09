#pragma once
#include <stdafx.h>
#include <vector>
#include <assert.h>
#include <memory>

#include "CudaKernel.h"
#include "libmmv/math/Vec3.h"
#include "solution/Solution.h"

extern "C" void CK_SolveDisplacement_launch(REAL* d_displacements, unsigned short* d_matConfigEquationIds, REAL* d_matConfigEquations, unsigned int numVertices);

class CK_SolveDisplacement : public CudaKernel {

public:

    CK_SolveDisplacement(Solution* sol);
    ~CK_SolveDisplacement();

    void launchKernel() override;

protected:

    bool canExecute() override;
    void freeCudaResources();

private:
    Solution* solution;

    unsigned short* d_matConfigEquationIds;
    REAL* d_displacements;
    REAL* d_matConfigEquations;

    void prepareInputs();

    void push_matConfigEquationIds();
    void push_displacements();
    void push_matConfigEquations();

    void pull_displacements();

    void serializeMaterialConfigurationEquations(void* destination);

};
