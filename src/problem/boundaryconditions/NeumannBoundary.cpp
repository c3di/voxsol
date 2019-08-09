#include "stdafx.h"
#include "NeumannBoundary.h"

NeumannBoundary::NeumannBoundary() :
    force(libmmv::Vec3<REAL>(0,0,0))
{

}

NeumannBoundary::NeumannBoundary(libmmv::Vec3<REAL>& s) :
    force(s)
{

}
NeumannBoundary::~NeumannBoundary() {

}

void NeumannBoundary::combine(NeumannBoundary& other) {
    force += other.force;
}
