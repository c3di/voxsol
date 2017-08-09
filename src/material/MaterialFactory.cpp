#include "stdafx.h"
#include "material/MaterialFactory.h"

MaterialFactory::MaterialFactory() {

}

MaterialFactory::~MaterialFactory() {

}

Material MaterialFactory::createMaterialWithProperties(REAL youngsModulus, REAL poissonRatio) {
    return Material(youngsModulus, poissonRatio, ++nextId);
}