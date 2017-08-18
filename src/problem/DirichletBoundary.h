#pragma once
#include <functional>

class DirichletBoundary {
public:
    
    enum Condition: unsigned char {
        NONE = 0,
        FIXED_X = 1,
        FIXED_Y = 2,
        FIXED_Z = 4,
        FIXED_ALL = FIXED_X | FIXED_Y | FIXED_Z
    };

    DirichletBoundary();
    DirichletBoundary(Condition condition);
    ~DirichletBoundary();

    char encodeAsChar() const;
    void combine(DirichletBoundary& other);
    bool hasFixedAxes() const;

    bool isXFixed() const;
    bool isYFixed() const;
    bool isZFixed() const;

    bool operator==(const DirichletBoundary& other) const {
        return fixed == other.fixed;
    }

    Condition fixed;
};

namespace std {
    template<>
    struct hash<DirichletBoundary>
    {
        std::size_t operator()(const DirichletBoundary& k) const
        {
            unsigned char condition = k.encodeAsChar();
            return std::hash<unsigned char>()(*reinterpret_cast<unsigned char*> (&condition));
        }
    };
}
