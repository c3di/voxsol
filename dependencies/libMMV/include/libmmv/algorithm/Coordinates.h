#pragma once

#include "libmmv/math/Vec2.h"
#include "libmmv/math/Vec3.h"

namespace libmmv
{
	unsigned int Flatten3D(Vec3ui index, Vec3ui size);
    Vec3ui       Unflatten3D(unsigned int index, Vec3ui size);
    Vec3ui       Unflatten3D(size_t index, Vec3ui size);
    unsigned int Flatten2D(Vec2ui index, Vec2ui size);

    inline unsigned int Flatten3D(Vec3ui index, Vec3ui size)
    {
        return index.z * size.y * size.x + index.y * size.x + index.x;
    }

    inline Vec3ui Unflatten3D(unsigned int index, Vec3ui size)
    {
        const unsigned int x = index % size.x;
        const unsigned int y = (index / size.x) % size.y;
        const unsigned int z = (index / (size.x*size.y)) % size.z;
        return Vec3ui( x, y, z );
    }

    inline Vec3ui Unflatten3D(size_t index, Vec3ui size)
    {
        const unsigned int x = index % size.x;
        const unsigned int y = (index / size.x) % size.y;
        const unsigned int z = (index / (size.x*size.y)) % size.z;
        return Vec3ui(x, y, z);
    }

    inline unsigned int Flatten2D(Vec2ui index, Vec2ui size)
    {
        return index.y * size.x + index.x;
    }

}