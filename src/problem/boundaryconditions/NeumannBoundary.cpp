#include "stdafx.h"
#include "NeumannBoundary.h"

NeumannBoundary::NeumannBoundary() :
    stress(libmmv::Vec3<REAL>(0,0,0))
{

}

NeumannBoundary::NeumannBoundary(libmmv::Vec3<REAL>& s) :
    stress(s)
{

}
NeumannBoundary::~NeumannBoundary() {

}

void NeumannBoundary::combine(NeumannBoundary& other) {
    stress += other.stress;
}
