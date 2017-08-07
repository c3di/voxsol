#pragma once
#include <stdafx.h>

class Material {

public:
    
    Material();
    Material(REAL eModul, REAL poissonRatio);
    Material(REAL eModul, REAL poissonRatio, unsigned char id);
    ~Material();

    const unsigned char m_id;
    const REAL m_lambda;
    const REAL m_mu;

    static Material EMPTY;
    static unsigned char NEXT_ID;
    
    bool operator==(const Material& other) const;

private:

    REAL calculateMu(REAL eModul, REAL poissonRatio);
    REAL calculateLambda(REAL eModul, REAL poissonRatio);

};
