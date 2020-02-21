#include <vector>

namespace DLHandsOn {
    class Shape {
    public:
        Shape();
        ~Shape();
    public:
        int getTotal() const;
        int getDims() const;
        void clear();
    private:
        std::vector<int> shape;
    };

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