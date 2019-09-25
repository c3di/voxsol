#include "stdafx.h"
#include "DisplacementBoundary.h"

DisplacementBoundary::DisplacementBoundary() :
    displacement(0,0,0)
{

}

DisplacementBoundary::DisplacementBoundary(libmmv::Vec3<REAL> disp) :
    displacement(disp)
{

}

DisplacementBoundary::~DisplacementBoundary() {

}

bool DisplacementBoundary::isNonZero() const {
    return displacement.x != 0 || displacement.y != 0 || displacement.z != 0;
}
