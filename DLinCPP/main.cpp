#include <iostream>
#include <vector>
#include "General.hpp"
#include "DataObject.hpp"
#include "DenseLayer.hpp"
#include "Utilities.hpp"

static void printData(const DLHandsOn::DataObject& data) {
    for (auto d : data.getData()) {
        std::cout << d << " ";
    }
    std::cout << std::endl;
}

static void testDenseLayer() {
    const int batch_size = 2;
    const int input_size = 3;
    const int output_size = 2;

    DLHandsOn::DataObject input(DLHandsOn::Shape({ batch_size, input_size }));
    input.fillValue(0.5f);
    DLHandsOn::DataObject output(DLHandsOn::Shape({ batch_size, output_size }));
    output.fillValue(0.0f);
    printData(input);
    printData(output);
    const std::vector<DLHandsOn::DataObject*> inputs{ &input };
    std::vector<DLHandsOn::DataObject*> outputs{ &output };

    DLHandsOn::DenseLayer denseLayer;
    denseLayer.setup(input_size, output_size, true);
    printData(denseLayer.getWeight());
    printData(denseLayer.getBias());

    denseLayer.forward(inputs, outputs);
    printData(input);
    printData(output);
}

int main() {
    testDenseLayer();
    return 0;
}