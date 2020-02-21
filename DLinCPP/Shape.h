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
} // namespace DLHandsOn