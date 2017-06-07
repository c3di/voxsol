#pragma once

#include "model/volume/Volume.h"

namespace ettention
{

    /* The class assumes that a volume is composed of slabs in z-direction. The class can then be used to compute the
       difference between the consecutive slabs. 
    */
    class DistanceBetweenConsecutiveSlabs
    {
    public:
        static float l2NormDistance( Volume* volume, unsigned int slabSize);
    };
}
