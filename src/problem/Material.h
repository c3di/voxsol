#pragma once
#include <stdafx.h>

class Material {

public:
    
    Material();
    Material(REAL rho, REAL eModul, REAL poissonRatio, REAL yieldStrength);
    Material(REAL rho, REAL eModul, REAL poissonRatio, REAL yieldStrength, unsigned char id);
    ~Material();

    const unsigned char m_id;
    const REAL m_lambda;
    const REAL m_mu;

    static Material EMPTY;
    static unsigned char NEXT_ID;

private:

    REAL calculateMu(REAL eModul, REAL poissonRatio);
    REAL calculateLambda(REAL eModul, REAL poissonRatio);

};
