#pragma once
#include "Layer.hpp"
#include "Utilities.hpp"

namespace DLHandsOn {
    class InputLayer : public Layer {
    public:
        InputLayer();
        virtual ~InputLayer();
    public:
        virtual std::vector<DataObject*> getAllWeights();
        virtual std::vector<DataObject*> getAllGrads();

        virtual void forward(const std::vector<DataObject*>& inputs,
                             std::vector<DataObject*>& outputs);
        virtual void backward(const std::vector<DataObject*>& inputs,
                              std::vector<DataObject*>& outputs,
                              std::vector<DataObject*>& input_diffs,
                              std::vector<DataObject*>& output_diffs);
    };

    InputLayer::InputLayer() {}

    InputLayer::~InputLayer() {}

    std::vector<DataObject*> InputLayer::getAllWeights() {
        // no weights
        std::vector<DataObject*> outputs;
        return outputs;
    }
    std::vector<DataObject*> InputLayer::getAllGrads() {
        // no grads
        std::vector<DataObject*> outputs;
        return outputs;
    }

    // forward inference
    void InputLayer::forward(const std::vector<DataObject*>& inputs, std::vector<DataObject*>& outputs) {
        assert(inputs.size() == outputs.size(), "Invalid input size and output size");

        for (size_t i = 0; i < inputs.size(); i++) {
            const DataObject& input = *inputs[i];
			DataObject& output = *outputs[i];

			const Shape& input_shape = input.getShape();
			const Shape& output_shape = output.getShape();

			const int input_size = input_shape.getSize(1);
			const int output_size = output_shape.getSize(1);

			// shape of input: N * input_size
			// shape of output: N * output_size			
			assert(input_shape == output_shape, "Invalid shape.");

			const DataType& input_data = input.getData();
			DataType& output_data = output.getData();
			output_data = input_data;
        }
    }

    // backward propagation
    void InputLayer::backward(const std::vector<DataObject*>& inputs, std::vector<DataObject*>& outputs, std::vector<DataObject*>& input_diffs, std::vector<DataObject*>& output_diffs) {
        // No need
    }
} // namespace DLHandsOn