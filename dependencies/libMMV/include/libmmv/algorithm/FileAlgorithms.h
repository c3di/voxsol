#pragma once
#include <string>

namespace libmmv
{

    class FileAlgorithms
    {
    public:
        static std::string normalizePath(const std::string &path);
    };
}