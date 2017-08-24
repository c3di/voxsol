#pragma once
#include "stdafx.h"

#ifdef USE_DOUBLE_PRECISION
#define VERTEX_ALIGN 32
#else
#define VERTEX_ALIGN 16
#endif

struct alignas(VERTEX_ALIGN) Vertex {
    REAL x;
    REAL y;
    REAL z;
    ConfigId materialConfigId;
};
