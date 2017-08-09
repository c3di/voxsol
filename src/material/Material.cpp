#include "stdafx.h"
#include "material/Material.h"

Material Material::EMPTY = Material(0, 0, 0);

Material::Material() :
    id(255),
    mu(0),
    lambda(0)
{

}

Material::Material(REAL eModul, REAL poissonRatio, unsigned char id) :
    lambda(calculateLambda(eModul, poissonRatio)),
    mu(calculateMu(eModul, poissonRatio)),
    id(id)
{
    
}

Material::Material(Material& other) : 
    lambda(other.lambda),
    mu(other.mu),
    id(other.id)
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
    return other.id == id;
}
