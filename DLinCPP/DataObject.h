#include <vector>
#include "Shape.h"

namespace DLHandsOn {
    class DataObject {
    public:
        typedef std::vector<float> DataType;
    public:
        DataObject();
        // DataObject(const Shape& newShape);
        ~DataObject();
    public: // member functions
        bool empty() const;
        void clear();
        void reshape(const Shape& newShape);
        Shape getShape() const;
        DataType& getData();
        void fillValue(const float val);
    private:
        DataType data;
        Shape shape;
    };
} // namespace DLHandsOn