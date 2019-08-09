#pragma once
#include "stdafx.h"
#include <functional>
#include "libmmv/math/Vec3.h"

class NeumannBoundary {
public:
    NeumannBoundary();
    NeumannBoundary(libmmv::Vec3<REAL>& force);
    ~NeumannBoundary();

    void combine(NeumannBoundary& other);

    bool operator==(const NeumannBoundary& other) const {
        return force == other.force;
    }

    libmmv::Vec3<REAL> force;
};

namespace std {

    // Hash combine function as implemented in the boost library
    template <class T>
    inline void hash_combine2(std::size_t& seed, const T& v)
    {
        std::hash<T> hasher;
        seed ^= hasher(v) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
    };

    template<>
    struct hash<NeumannBoundary>
    {
        std::size_t operator()(const NeumannBoundary& k) const
        {
            std::hash<REAL> hasher;
            size_t hash = hasher(k.stress.x);
            hash_combine2(hash, hasher(k.stress.y));
            
            hash_combine2(hash, hasher(k.stress.z));
            return hash;
        }
    };
}
