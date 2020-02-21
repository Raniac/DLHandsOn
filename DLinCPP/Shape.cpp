#include "Shape.h"

namespace DLHandsOn {
    Shape::Shape() : shape() {}

    Shape::~Shape() {}

    int Shape::getTotal() const {
        if (shape.empty()) return 0;
        int total = 1;
        for (size_t i = 0; i < shape.size(); i++) {
            total *= shape[i];
        }
        return total;
    }

    int Shape::getDims() const { return (int)shape.size(); }

    void Shape::clear() { shape.clear(); }
} // namespace DLHandsOn