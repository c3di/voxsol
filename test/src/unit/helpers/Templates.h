#include "problem/Material.h"

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

};

Material Templates::Mat::EMPTY = Material(0, 0, 0, 0, 200);
Material Templates::Mat::FEMUR = Material(1360, 124000000, 0.4, 0.0, 201);
Material Templates::Mat::TIBIA = Material(1360, 174000000, 0.4, 0.0, 202);
Material Templates::Mat::STEEL = Material(7830, 210000000000, 0.3, 203);
Material Templates::Mat::TITANIUM = Material(4540, 110000000000, 0.33, 204);
