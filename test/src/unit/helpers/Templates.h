#pragma once
#include "material/MaterialDictionary.h"
#include "problem/DiscreteProblem.h"

class MatStore {
public:
    MatStore() :
        EMPTY(Material::EMPTY),
        FEMUR(124000000, 0.4, 201),
        TIBIA(174000000, 0.4, 202),
        STEEL(210000000000, 0.3, 203),
        TITANIUM(110000000000, 0.33, 204)
    {
        DICTIONARY.addMaterial(FEMUR);
        DICTIONARY.addMaterial(TIBIA);
        DICTIONARY.addMaterial(STEEL);
        DICTIONARY.addMaterial(TITANIUM);
    }

    MaterialDictionary DICTIONARY;
    Material EMPTY;
    Material FEMUR;
    Material TIBIA;
    Material STEEL;
    Material TITANIUM;
};

class Templates {
public:

    static MatStore Mat;

    class Problem {
    public:
        static DiscreteProblem STEEL_2_2_2() {
            ettention::Vec3ui size(2, 2, 2);
            ettention::Vec3<REAL> voxelSize(1, 1, 1);
            DiscreteProblem problem(size, voxelSize, &Templates::Mat.DICTIONARY);

            for (int i = 0; i < 8; i++) {
                problem.setMaterial(i, Templates::Mat.STEEL.id);
            }

            return problem;
        }
    };

};

