#pragma once
#include "material/Material.h"
#include <functional>
#include <vector>

#pragma pack(push, 1)
struct MaterialConfiguration { 
    unsigned char ids[8]; 

    MaterialConfiguration(const std::vector<Material*>* mats) {
        ids[0] = mats->at(0)->id;
        ids[1] = mats->at(1)->id;
        ids[2] = mats->at(2)->id;
        ids[3] = mats->at(3)->id;
        ids[4] = mats->at(4)->id;
        ids[5] = mats->at(5)->id;
        ids[6] = mats->at(6)->id;
        ids[7] = mats->at(7)->id;
    }

    bool operator==(const MaterialConfiguration& other) const {
        return ids[0] == other[0] && ids[1] == other[1] && ids[2] == other[2] && ids[3] == other[3] &&
            ids[4] == other[4] && ids[5] == other[5] && ids[6] == other[6] && ids[7] == other[7];
    }

    bool operator!=(const MaterialConfiguration& other) const {
        return !(ids[0] == other[0] && ids[1] == other[1] && ids[2] == other[2] && ids[3] == other[3] &&
            ids[4] == other[4] && ids[5] == other[5] && ids[6] == other[6] && ids[7] == other[7]);
    }

    const unsigned char& operator[](int index) const {
        return ids[index];
    }

};
#pragma pack(pop)

namespace std {

    template<>
    struct hash<MaterialConfiguration> 
    {
        std::size_t operator()(const MaterialConfiguration& k) const 
        {
            return std::hash<unsigned long long>()(*reinterpret_cast<const unsigned long long*> (&k));
        }
    };
}