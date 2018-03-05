#pragma once
#include "stdafx.h"

#ifdef USE_DOUBLE_PRECISION
#define VERTEX_ALIGN 32
#else
#define VERTEX_ALIGN 16
#endif

struct alignas(VERTEX_ALIGN) Vertex {
    volatile REAL x = 0;
    volatile REAL y = 0;
    volatile REAL z = 0;
    ConfigId materialConfigId = 0;
};
