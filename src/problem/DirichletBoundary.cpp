#include "stdafx.h"
#include "DirichletBoundary.h"

DirichletBoundary::DirichletBoundary() :
    fixed(Condition::NONE)
{

}

DirichletBoundary::DirichletBoundary(Condition condition) :
    fixed(condition)
{

}
DirichletBoundary::~DirichletBoundary() {

}

char DirichletBoundary::encodeAsChar() const {
    return (char)fixed;
}

void DirichletBoundary::combine(DirichletBoundary& other) {
    fixed = (Condition) (fixed | other.encodeAsChar());
}

bool DirichletBoundary::hasFixedAxes() const {
    return fixed != Condition::NONE;
}

bool DirichletBoundary::isXFixed() const {
    return (fixed & Condition::FIXED_X) != 0;
}

bool DirichletBoundary::isYFixed() const {
    return (fixed & Condition::FIXED_Y) != 0;
}

bool DirichletBoundary::isZFixed() const {
    return (fixed & Condition::FIXED_Z) != 0;
}
