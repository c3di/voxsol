#pragma once
#include <iostream>
#include "math/Matrix3x3.h"
#include "math/Matrix4x4.h"

namespace ettention
{
    std::istream &operator>>(std::istream& is, Matrix3x3& value);
    std::ostream &operator<<(std::ostream& os, Matrix3x3 value);

    std::istream &operator>>(std::istream& is, Matrix4x4& value);
    std::ostream &operator<<(std::ostream& os, Matrix4x4 value);
}
