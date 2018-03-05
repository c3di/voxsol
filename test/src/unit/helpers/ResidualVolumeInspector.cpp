#include "helpers/ResidualVolumeInspector.h"
#include "problem/ProblemFragment.h"
#include "problem/DiscreteProblem.h"
#include <sstream>

ResidualVolumeInspector::ResidualVolumeInspector(DiscreteProblem & problem) :
    ResidualVolume(problem)
{
}


void ResidualVolumeInspector::updateAllLevelsAboveZero() {
    libmmv::Vec3ui from(0, 0, 0);
    updatePyramid(1, from, levelZeroSize / 2);
}
