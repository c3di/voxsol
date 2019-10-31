#pragma once
#include <functional>
#include "libmmv/math/Vec3.h"
#include "stdafx.h"

class DisplacementBoundary {
public:
    
    DisplacementBoundary();
    DisplacementBoundary(libmmv::Vec3<REAL> disp);
    ~DisplacementBoundary();

    bool isNonZero() const;

    bool operator==(const DisplacementBoundary& other) const {
        return displacement == other.displacement;
    }

    libmmv::Vec3<REAL> displacement;
};

namespace std {

    // Hash combine function as implemented in the boost library
    template <class T>
    inline void hash_combine3(std::size_t& seed, const T& v)
    {
        std::hash<T> hasher;
        seed ^= hasher(v) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
    };

    template<>
    struct hash<DisplacementBoundary>
    {
        std::size_t operator()(const DisplacementBoundary& k) const
        {
            std::hash<REAL> hasher;
            size_t hash = hasher(k.displacement.x);
            hash_combine3(hash, hasher(k.displacement.y));

            hash_combine3(hash, hasher(k.displacement.z));
            return hash;
        }
    };
}
