#include <vector>
// #include "DataObject.h"
#include "Shape.hpp"

namespace DLHandsOn {
    typedef std::vector<float> DataType;

    class DataObject {
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

    DataObject::DataObject() : data(), shape() {}

    // DataObject::DataObject(const Shape& newShape) {}

    DataObject::~DataObject() {}

    bool DataObject::empty() const { return shape.getTotal() == 0; }

    void DataObject::clear() {
        shape.clear();
        data.clear();
    }

    void DataObject::reshape(const Shape& newShape) {
        shape = newShape;
        data.resize(shape.getTotal());
        // fill zero?
    }

    Shape DataObject::getShape() const { return shape; }

    DataType& DataObject::getData() { return data; }

    void DataObject::fillValue(const float val) {
        std::fill(data.begin(), data.end(), val);
    }
}