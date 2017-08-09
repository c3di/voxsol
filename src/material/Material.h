#pragma once
#include <stdafx.h>

class Material {

public:
    
    Material();
    Material(REAL eModul, REAL poissonRatio, unsigned char id);
    Material(Material& other);
    ~Material();

    const unsigned char id;
    const REAL lambda;
    const REAL mu;

    static Material EMPTY;
    static unsigned char NEXT_ID;
    
    bool operator==(const Material& other) const;

private:

    REAL calculateMu(REAL eModul, REAL poissonRatio);
    REAL calculateLambda(REAL eModul, REAL poissonRatio);

};