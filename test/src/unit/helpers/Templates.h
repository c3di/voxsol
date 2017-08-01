#pragma once
#include "problem/Material.h"
#include "problem/DiscreteProblem.h"

class Templates {
public:

    class Mat {
    public:
        static Material EMPTY;
        static Material FEMUR;
        static Material TIBIA;
        static Material STEEL;
        static Material TITANIUM;
    };

    class Problem {
    public:
        static DiscreteProblem STEEL_2_2_2() {
            ettention::Vec3ui size(2, 2, 2);
            ettention::Vec3<REAL> voxelSize(1, 1, 1);
            DiscreteProblem problem(size, voxelSize);

            for (int i = 0; i < 8; i++) {
                problem.setMaterial(i, Templates::Mat::STEEL);
            }

            return problem;
        }
    };

};

