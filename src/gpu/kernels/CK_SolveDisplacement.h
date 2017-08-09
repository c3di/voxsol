#pragma once
#include <stdafx.h>
#include <vector>
#include <assert.h>
#include <memory>

#include "CudaKernel.h"
#include "libmmv/math/Vec3.h"
#include "solution/Solution.h"

extern "C" void CK_SolveDisplacement_launch(REAL* d_displacements, unsigned short* d_signatureIds, REAL* d_fragmentSignatures, unsigned int numVertices);

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

    unsigned short* d_signatureIds;
    REAL* d_displacements;
    REAL* d_fragmentSignatures;

    void prepareInputs();

    void push_signatureIds();
    void push_displacements();
    void push_fragmentSignatures();

    void pull_displacements();

    void serializeFragmentSignatures(void* destination);

};
