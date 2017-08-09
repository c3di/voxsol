#pragma once
#include <unordered_map>
#include "material/Material.h"

class MaterialDictionary {
public:
    MaterialDictionary();
    ~MaterialDictionary();

    void addMaterial(Material& material);
    Material* getMaterialById(unsigned char materialId);
    bool contains(unsigned char materialId);


protected:
    std::unordered_map<unsigned char, Material> materials;

};