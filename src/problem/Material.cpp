#include "stdafx.h"
#include "problem/Material.h"

Material::Material(unsigned int id, double rho, double eModul, double poissonRatio, double yieldStrength) :
    m_lambda(calculateLambda(eModul, poissonRatio)),
    m_mu(calculateMu(eModul, poissonRatio)),
    m_id(id)
{

}
Material::~Material() {

}

double Material::calculateMu(double eModul, double poissonRatio) {
    return eModul / (2.0 * (1.0 + poissonRatio));
}

double Material::calculateLambda(double eModul, double poissonRatio) {
    double divisor = (1.0 + poissonRatio) * (1.0 - 2.0 * poissonRatio);
    return (eModul * poissonRatio) / divisor;
}
