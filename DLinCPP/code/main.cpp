#include <iostream>
#include <vector>
#include "General.hpp"
#include "DataObject.hpp"
#include "InputLayer.hpp"
#include "DenseLayer.hpp"
#include "ActivationLayer.hpp"
#include "LossFunction.hpp"
#include "Network.hpp"
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

static void demoTrain() {
    const int batch_size = 2;
    const int input_size = 3;
    const int dense_in_size = 3;
    const int dense_out_size = 3;
    const int output_size = 2;

    // data definition
    //input wrapper
	std::vector<DLHandsOn::DataObject*> inputs;
	DLHandsOn::DataObject input(DLHandsOn::Shape{batch_size, input_size});
    input.fillValue(0.5f);
	inputs.push_back(&input);

	//predicts wrapper
	std::vector<DLHandsOn::DataObject*> outputs;
	DLHandsOn::DataObject output(DLHandsOn::Shape{ batch_size, output_size });
    output.fillValue(0.0f);
	outputs.push_back(&output);

	//output wrapper
	std::vector<DLHandsOn::DataObject*> ground_truths;
	DLHandsOn::DataObject ground_truth(DLHandsOn::Shape{ batch_size, output_size });
    ground_truth.fillValue(0.4f);
	ground_truths.push_back(&ground_truth);

    printData("Input Data Before Training:", input);
    printData("Output Data Before Training:", output);
    printData("Ground Truth Before Training:", ground_truth);

    // network definition
    DLHandsOn::Network network;

    DLHandsOn::InputLayer input_layer;
    network = network.addLayer(&input_layer);

    DLHandsOn::DenseLayer dense_layer;
    dense_layer.setup(dense_in_size, dense_out_size, true);
    dense_layer.setPhase(DLHandsOn::Layer::Phase::Train);
    network = network.addLayer(&dense_layer);

    DLHandsOn::SigmoidLayer sigmoid_layer;
    network = network.addLayer(&sigmoid_layer);

    DLHandsOn::MSELoss loss_funciton;
    network.setLoss(&loss_funciton);

    // training
    const int num_epochs = 2;
    for (size_t i = 0; i < num_epochs; i++) {
        for (size_t j = 0; j < batch_size; j++) {

			//forward
			network.forward(inputs, outputs);
			//backward
			network.backward(inputs, ground_truths);
			//update weights
			network.updateWeights();

			//get loss
			const float loss = network.getLoss(ground_truths, outputs);
			std::cout << "Epoch[" << i <<"/" << num_epochs <<  "] " <<
				"Batch[" << j << "/" << batch_size << "] " << 
				"loss: " << loss << std::endl;
        }
    }
}

int main() {
    std::cout << "Deep Learning Hands-On!" << std::endl;
    demoTrain();
    return 0;
}