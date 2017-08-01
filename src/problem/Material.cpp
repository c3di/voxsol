#include "stdafx.h"
#include "problem/Material.h"

Material Material::EMPTY = Material(0, 0, 0, 0, 0);
unsigned char Material::NEXT_ID = 1;

Material::Material() :
    m_id(255),
    m_mu(0),
    m_lambda(0)
{

}

Material::Material(REAL rho, REAL eModul, REAL poissonRatio, REAL yieldStrength) :
    m_lambda(calculateLambda(eModul, poissonRatio)),
    m_mu(calculateMu(eModul, poissonRatio)),
    m_id(NEXT_ID)
{
    NEXT_ID++;
}

Material::Material(REAL rho, REAL eModul, REAL poissonRatio, REAL yieldStrength, unsigned char id) :
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

bool Material::operator==(const Material& other) const
{
    return other.m_id == m_id;
}
