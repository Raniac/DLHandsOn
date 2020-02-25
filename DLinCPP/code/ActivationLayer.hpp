#pragma once
#include <cmath>
#include "Layer.hpp"
#include "Utilities.hpp"

namespace DLHandsOn
{
	// activation functions

	// Sigmoid Layer
	// y = 1.0 / (1.0 + e^(-x))
	class SigmoidLayer : public Layer {
	public:
        // get all weights
        virtual std::vector<DataObject*> getAllWeights();

        // get all grads
        virtual std::vector<DataObject*> getAllGrads();

		// forward function
        virtual void forward(const std::vector<DataObject*>& inputs,
                             std::vector<DataObject*>& outputs);

        // backward function
        virtual void backward(const std::vector<DataObject*>& inputs,
                              std::vector<DataObject*>& outputs,
                              std::vector<DataObject*>& input_diffs,
                              std::vector<DataObject*>& output_diffs);
	};

    std::vector<DataObject*> SigmoidLayer::getAllWeights() {
        std::vector<DataObject*> all_weights;
        return all_weights;
    }

    std::vector<DataObject*> SigmoidLayer::getAllGrads() {
        std::vector<DataObject*> all_grads;
        return all_grads;
    }

    void SigmoidLayer::forward(const std::vector<DataObject*>& inputs, std::vector<DataObject*>& outputs) {
        // forward
        // y = 1.0 / (1.0 + e^(-x))
        assert(inputs.size() == outputs.size(), "Invalid input size and output size.");
        assert(inputs.size() > 0, "Invalid input size, less than zero.");

        for (size_t i = 0; i < inputs.size(); i++) {
            const DataObject& input = *inputs[i];
            DataObject& output = *outputs[i];
            const DataType& input_data = input.getData();
            DataType& output_data = output.getData();

            for (size_t j = 0; j < input_data.size(); j++) {
                output_data[j] = 1.0f / (1.0f + std::exp(-input_data[j]));
            }
        }
    }

    void SigmoidLayer::backward(const std::vector<DataObject*>& inputs, std::vector<DataObject*>& outputs, std::vector<DataObject*>& input_diffs, std::vector<DataObject*>& output_diffs) {
        // backward
        // dy/dx = y * (1 - y)
        // dloss/dx = dloss/dy * dy/dx = dloss/dy * y * (1 - y)
        assert(inputs.size() == outputs.size(), "Invalid inputs and outputs size.");
        assert(input_diffs.size() == inputs.size(), "Invalid inputs and diffs size.");
        assert(output_diffs.size() == outputs.size(), "Invalid diffs and outputs size.");
        assert(inputs.size() > 0, "Invalid inputs size.");

        for (size_t i = 0; i < input_diffs.size(); i++) {
            DataObject& input_diff = *input_diffs[i];
            const DataObject& output_diff = *output_diffs[i];
            const DataObject& input = *inputs[i];
            const DataObject& output = *outputs[i];

            const DataType& input_data = input.getData();
            const DataType& output_data = output.getData();
            input_diff.fillValue(0.0f);
            DataType& input_diff_data = input_diff.getData();
			const DataType& output_diff_data = output_diff.getData();

            for (size_t j = 0; j < input_diff_data.size(); j++) {
                input_diff_data[j] = output_diff_data[j] * (output_data[j] * (1.0f - output_data[j]));
            }
        }
    }
} // namespace DLHandsOn