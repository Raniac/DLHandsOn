#include <iostream>
#include "Utilities.h"

namespace DLHandsOn {
    void assert(const bool condition, const char* message, ...) {
        if (!condition) {
            std::cout << message << std::endl;
        }
    }
} // DLHandsOn