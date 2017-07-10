#include <functional>

#pragma pack(push, 1)
struct ProblemFragmentKey { 
    unsigned char ids[8]; 

    ProblemFragmentKey(const std::vector<Material*>* mats) {
        ids[0] = mats->at(0)->m_id;
        ids[1] = mats->at(1)->m_id;
        ids[2] = mats->at(2)->m_id;
        ids[3] = mats->at(3)->m_id;
        ids[4] = mats->at(4)->m_id;
        ids[5] = mats->at(5)->m_id;
        ids[6] = mats->at(6)->m_id;
        ids[7] = mats->at(7)->m_id;
    }

    bool operator==(const ProblemFragmentKey& other) const {
        return ids[0] == other[0] && ids[1] == other[1] && ids[2] == other[2] && ids[3] == other[3] &&
            ids[4] == other[4] && ids[5] == other[5] && ids[6] == other[6] && ids[7] == other[7];
    }

    bool operator!=(const ProblemFragmentKey& other) const {
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
    struct hash<ProblemFragmentKey> 
    {
        std::size_t operator()(const ProblemFragmentKey& k) const 
        {
            return std::hash<unsigned long long>()(*reinterpret_cast<const unsigned long long*> (&k));
        }
    };
}