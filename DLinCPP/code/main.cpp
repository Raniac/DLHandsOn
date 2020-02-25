#include <iostream>
#include <vector>
#include "General.hpp"
#include "DataObject.hpp"
#include "DenseLayer.hpp"
#include "ActivationLayer.hpp"
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
    DLHandsOn::DataObject input_diff(input.getShape());
    input_diff.fillValue(0.0f);
    DLHandsOn::DataObject output_diff(output.getShape());
    output_diff.fillValue(1.0f);

    printData("Input Data Before Dense:", input);
    printData("Output Data Before Dense:", output);
    const std::vector<DLHandsOn::DataObject*> inputs = { &input };
    std::vector<DLHandsOn::DataObject*> outputs = { &output };
    std::vector<DLHandsOn::DataObject*> input_diffs = { &input_diff };
    std::vector<DLHandsOn::DataObject*> output_diffs = { &output_diff };

    DLHandsOn::DenseLayer denseLayer;
    denseLayer.setup(input_size, output_size, true);
    denseLayer.setPhase(DLHandsOn::Layer::Phase::Train);
    printData("Initialized Weights:", denseLayer.getWeights());
    printData("Initialized Bias:", denseLayer.getBias());

    denseLayer.forward(inputs, outputs);
    printData("Input Data After Dense:", input);
    printData("Output Data After Dense:", output);

    denseLayer.backward(inputs, outputs, input_diffs, output_diffs);
    printData("Input Diff After Dense Back Propagation:", input_diff);
}

static void testSigmoidLayer() {
    const int batch_size = 2;
    const int input_size = 3;
    const int output_size = 3;

    DLHandsOn::DataObject input(DLHandsOn::Shape({ batch_size, input_size }));
    input.fillValue(0.5f);
    DLHandsOn::DataObject output(DLHandsOn::Shape({ batch_size, output_size }));
    output.fillValue(0.0f);
    DLHandsOn::DataObject input_diff(input.getShape());
    input_diff.fillValue(0.0f);
    DLHandsOn::DataObject output_diff(output.getShape());
    output_diff.fillValue(1.0f);

    printData("Input Data Before Dense:", input);
    printData("Output Data Before Dense:", output);
    const std::vector<DLHandsOn::DataObject*> inputs = { &input };
    std::vector<DLHandsOn::DataObject*> outputs = { &output };
    std::vector<DLHandsOn::DataObject*> input_diffs = { &input_diff };
    std::vector<DLHandsOn::DataObject*> output_diffs = { &output_diff };

    DLHandsOn::SigmoidLayer sigmoidLayer;
    sigmoidLayer.setPhase(DLHandsOn::Layer::Phase::Train);

    sigmoidLayer.forward(inputs, outputs);
    printData("Input Data After Dense:", input);
    printData("Output Data After Dense:", output);

    sigmoidLayer.backward(inputs, outputs, input_diffs, output_diffs);
    printData("Input Diff After Dense Back Propagation:", input_diff);
}

int main() {
    std::cout << "Deep Learning Hands-On!" << std::endl;
    testSigmoidLayer();
    return 0;
}