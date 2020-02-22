#include <iostream>
#include <vector>
#include "General.hpp"
#include "DataObject.hpp"
#include "DenseLayer.hpp"
#include "Utilities.hpp"

static void printData(std::string data_info, const DLHandsOn::DataObject& data) {
    std::cout << data_info << std::endl << "[ ";
    for (auto d : data.getData()) {
        std::cout << d << " ";
    }
    std::cout << "]" << std::endl;
}

static void testDenseLayer() {
    const int batch_size = 2;
    const int input_size = 3;
    const int output_size = 2;

    DLHandsOn::DataObject input(DLHandsOn::Shape({ batch_size, input_size }));
    input.fillValue(0.5f);
    DLHandsOn::DataObject output(DLHandsOn::Shape({ batch_size, output_size }));
    output.fillValue(0.0f);
    printData("Input Data Before Dense:", input);
    printData("Output Data Before Dense:", output);
    const std::vector<DLHandsOn::DataObject*> inputs{ &input };
    std::vector<DLHandsOn::DataObject*> outputs{ &output };

    DLHandsOn::DenseLayer denseLayer;
    denseLayer.setup(input_size, output_size, true);
    printData("Initialized Weights:", denseLayer.getWeight());
    printData("Initialized Bias:", denseLayer.getBias());

    denseLayer.forward(inputs, outputs);
    printData("Input Data After Dense:", input);
    printData("Output Data After Dense:", output);
}

int main() {
    std::cout << "Deep Learning Hands-On!" << std::endl;
    testDenseLayer();
    return 0;
}