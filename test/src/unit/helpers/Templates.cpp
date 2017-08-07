#include "Templates.h"
#include "problem/DiscreteProblem.h"

Material Templates::Mat::EMPTY = Material::EMPTY;
Material Templates::Mat::FEMUR = Material(124000000, 0.4, 201);
Material Templates::Mat::TIBIA = Material(174000000, 0.4, 202);
Material Templates::Mat::STEEL = Material(210000000000, 0.3, 203);
Material Templates::Mat::TITANIUM = Material(110000000000, 0.33, 204);

