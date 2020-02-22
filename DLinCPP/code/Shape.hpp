#pragma once
#include <vector>
#include "Utilities.hpp"

namespace DLHandsOn {
    class Shape {
    public:
        Shape();
        Shape(const std::initializer_list<int> items);
        ~Shape();
    public:
        int getTotal() const;
        int getSize(const int index) const;
        int getDims() const;
        void clear();
        bool operator==(const Shape& other) const;
    private:
        std::vector<int> shape;
    };

    Shape::Shape() : shape() {}

    Shape::Shape(const std::initializer_list<int> val) { shape = val; }

    Shape::~Shape() {}

    int Shape::getTotal() const {
        if (shape.empty()) return 0;
        int total = 1;
        for (size_t i = 0; i < shape.size(); i++) {
            total *= shape[i];
        }
        return total;
    }

    int Shape::getSize(const int index) const {
		assert(index >= 0 && index < shape.size(), "Invalid index.");
		return shape[index];
	}

    int Shape::getDims() const { return (int)shape.size(); }

    void Shape::clear() { shape.clear(); }

    bool Shape::operator==(const Shape& other) const {
		return this->shape == other.shape;
	}
} // namespace DLHandsOn