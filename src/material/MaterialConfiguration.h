#pragma once
#include "problem/ProblemFragment.h"
#include "problem/boundaryconditions/DirichletBoundary.h"
#include "problem/boundaryconditions/NeumannBoundary.h"
#include "problem/boundaryconditions/DisplacementBoundary.h"
#include "material/Material.h"
#include <vector>

#pragma pack(push, 1)
struct MaterialConfiguration { 
    unsigned char ids[8]; 
    DirichletBoundary dirichletBoundaryCondition;
    NeumannBoundary neumannBoundaryCondition;
    DisplacementBoundary displacementBoundaryCondition;

    MaterialConfiguration(const ProblemFragment* fragment) {
        const std::vector<Material*>* mats = fragment->getMaterials();
        ids[0] = mats->at(0)->id;
        ids[1] = mats->at(1)->id;
        ids[2] = mats->at(2)->id;
        ids[3] = mats->at(3)->id;
        ids[4] = mats->at(4)->id;
        ids[5] = mats->at(5)->id;
        ids[6] = mats->at(6)->id;
        ids[7] = mats->at(7)->id;
        dirichletBoundaryCondition = fragment->getDirichletBoundaryCondition();
        neumannBoundaryCondition = fragment->getNeumannBoundaryCondition();
        displacementBoundaryCondition = fragment->getDisplacementBoundaryCondition();
    }

    MaterialConfiguration(const std::vector<Material*>* mats) {
        ids[0] = mats->at(0)->id;
        ids[1] = mats->at(1)->id;
        ids[2] = mats->at(2)->id;
        ids[3] = mats->at(3)->id;
        ids[4] = mats->at(4)->id;
        ids[5] = mats->at(5)->id;
        ids[6] = mats->at(6)->id;
        ids[7] = mats->at(7)->id;
        dirichletBoundaryCondition = DirichletBoundary(DirichletBoundary::NONE);
    }

    //Note: Displacement boundaries are not considered here as they have to be handled as a special case in Solution.cpp where they are 
    // given the EMPTY_MATERIALS id to prevent the vertex from ever being updated
    bool operator==(const MaterialConfiguration& other) const {
        return ids[0] == other[0] && ids[1] == other[1] && ids[2] == other[2] && ids[3] == other[3] &&
            ids[4] == other[4] && ids[5] == other[5] && ids[6] == other[6] && ids[7] == other[7] &&
            dirichletBoundaryCondition == other.dirichletBoundaryCondition &&
            neumannBoundaryCondition == other.neumannBoundaryCondition;
    }

    bool operator!=(const MaterialConfiguration& other) const {
        return !(*this == other);
    }

    const unsigned char& operator[](int index) const {
        return ids[index];
    }

    bool isVoidConfiguration() const {
        return ids[0] == 0 && ids[1] == 0 && ids[2] == 0 && ids[3] == 0 &&
            ids[4] == 0 && ids[5] == 0 && ids[6] == 0 && ids[7] == 0;
    }

};
#pragma pack(pop)

namespace std {

    template<>
    struct hash<MaterialConfiguration> 
    {
        std::size_t operator()(const MaterialConfiguration& k) const 
        {
            std::size_t hashValue = std::hash<unsigned long long>()(*reinterpret_cast<const unsigned long long*> (&k));
            hash_combine(hashValue, k.dirichletBoundaryCondition);
            hash_combine(hashValue, k.neumannBoundaryCondition);
            return hashValue;
        }
    };

}
