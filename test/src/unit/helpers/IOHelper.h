#pragma once

#include <experimental/filesystem>

class IOHelper {
public:

    static std::string getAbsolutePathToFile(const std::string& relativePath) {
        namespace fs = std::experimental::filesystem;
        // or namespace fs = boost::filesystem;
        auto baseDir = fs::current_path();
        while (baseDir.has_parent_path())
        {
            auto combinePath = baseDir / relativePath;
            if (fs::exists(combinePath))
            {
                return combinePath.string();
            }
            baseDir = baseDir.parent_path();
        }
        throw std::runtime_error("File not found!");
    }
};