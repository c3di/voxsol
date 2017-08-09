#include "stdafx.h"
#include "material/Material.h"

Material Material::EMPTY = Material(0, 0, 0);

Material::Material() :
    m_id(255),
    m_mu(0),
    m_lambda(0)
{

}

Material::Material(REAL eModul, REAL poissonRatio, unsigned char id) :
    m_lambda(calculateLambda(eModul, poissonRatio)),
    m_mu(calculateMu(eModul, poissonRatio)),
    m_id(id)
{
    
}

Material::Material(Material& other) : 
    m_lambda(other.m_lambda),
    m_mu(other.m_mu),
    m_id(other.m_id)
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
