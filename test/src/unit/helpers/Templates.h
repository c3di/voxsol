#pragma once
#include "stdafx.h"
#include "material/MaterialDictionary.h"
#include "problem/DiscreteProblem.h"

class MatStore {
public:
    MatStore() :
        EMPTY(Material::EMPTY),
        FEMUR(asREAL(124000000), asREAL(0.4), 201),
        TIBIA(asREAL(174000000), asREAL(0.4), 202),
        STEEL(asREAL(210000000000), asREAL(0.3), 203),
        TITANIUM(asREAL(110000000000), asREAL(0.33), 204)
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
            libmmv::Vec3ui size(2, 2, 2);
            libmmv::Vec3<REAL> voxelSize(1, 1, 1);
            DiscreteProblem problem(size, voxelSize, &Templates::Mat.DICTIONARY);

            for (int i = 0; i < 8; i++) {
                problem.setMaterial(i, Templates::Mat.STEEL.id);
            }

            return problem;
        }

        static DiscreteProblem STEEL_3_3_3() {
            libmmv::Vec3ui size(3, 3, 3);
            libmmv::Vec3<REAL> voxelSize(1, 1, 1);
            DiscreteProblem problem(size, voxelSize, &Templates::Mat.DICTIONARY);

            for (int i = 0; i < 27; i++) {
                problem.setMaterial(i, Templates::Mat.STEEL.id);
            }

            return problem;
        }

        static DiscreteProblem STEEL(libmmv::Vec3ui& size) {
            libmmv::Vec3<REAL> voxelSize(1, 1, 1);
            int numVoxels = size.x * size.y * size.z;
            DiscreteProblem problem(size, voxelSize, &Templates::Mat.DICTIONARY);

            for (int i = 0; i < numVoxels; i++) {
                problem.setMaterial(i, Templates::Mat.STEEL.id);
            }

            return problem;
        }

        static DiscreteProblem STEEL(libmmv::Vec3ui& discretization, libmmv::Vec3<REAL>& dimensions) {
            libmmv::Vec3<REAL> voxelSize(dimensions.x / discretization.x, dimensions.y / discretization.y, dimensions.z / discretization.z);
            int numVoxels = discretization.x * discretization.y * discretization.z;
            DiscreteProblem problem(discretization, voxelSize, &Templates::Mat.DICTIONARY);

            for (int i = 0; i < numVoxels; i++) {
                problem.setMaterial(i, Templates::Mat.STEEL.id);
            }

            return problem;
        }
    };

};

