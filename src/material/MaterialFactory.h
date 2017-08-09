#pragma once
#include "material/Material.h"

class MaterialFactory {
public:
    MaterialFactory();
    ~MaterialFactory();

    Material createMaterialWithProperties(REAL youngsModulus, REAL poissonRatio);

protected:
    unsigned char nextId;

};