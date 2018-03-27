#pragma once
#include "gpu/sampling/ResidualVolume.h"

class ProblemFragment;

class ResidualVolumeInspector : public ResidualVolume {
public:
    ResidualVolumeInspector(DiscreteProblem* problem);

    void updateAllLevelsAboveZero();

};

