#include "stdafx.h"
#include "problem/Material.h"

Material Material::EMPTY = Material(65, 0, 0, 0, 0);

Material::Material(char id, REAL rho, REAL eModul, REAL poissonRatio, REAL yieldStrength) :
    m_lambda(calculateLambda(eModul, poissonRatio)),
    m_mu(calculateMu(eModul, poissonRatio)),
    m_id(id)
{

}
Material::~Material() {

}

REAL Material::calculateMu(REAL eModul, REAL poissonRatio) {
    return eModul / (2.0 * (1.0 + poissonRatio));
}

REAL Material::calculateLambda(REAL eModul, REAL poissonRatio) {
    REAL divisor = (1.0 + poissonRatio) * (1.0 - 2.0 * poissonRatio);
    return (eModul * poissonRatio) / divisor;
}
