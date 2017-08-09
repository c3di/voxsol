#include "stdafx.h"
#include "MaterialDictionary.h"

MaterialDictionary::MaterialDictionary() {
    materials.emplace(0, Material::EMPTY);
}

MaterialDictionary::~MaterialDictionary() {

}

Material* MaterialDictionary::getMaterialById(unsigned char materialId) {
    if (materials.count(materialId) == 0) {
        return nullptr;
    }
    return &materials[materialId];
}

void MaterialDictionary::addMaterial(Material& material) {
    assert(materials.count(material.id) <= 0);
    materials.emplace(material.id, material);
}

bool MaterialDictionary::contains(unsigned char materialId) {
    return materials.count(materialId) > 0;
}