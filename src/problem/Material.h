#pragma once
#include <stdafx.h>

class Material {

public:
    
    Material(char id, REAL rho, REAL eModul, REAL poissonRatio, REAL yieldStrength);
    ~Material();

    const char m_id;
    const REAL m_lambda;
    const REAL m_mu;

    static Material EMPTY;

private:

    REAL calculateMu(REAL eModul, REAL poissonRatio);
    REAL calculateLambda(REAL eModul, REAL poissonRatio);

};
