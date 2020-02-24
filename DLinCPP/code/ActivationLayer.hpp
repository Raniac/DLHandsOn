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

    std::vector<DataObject*> SigmoidLayer::getAllGrads() {
        std::vector<DataObject*> all_grads;
        return all_grads;
    }

    void SigmoidLayer::forward(const std::vector<DataObject*>& inputs, std::vector<DataObject*>& outputs) {
        // forward
        // y = 1.0 / (1.0 + e^(-x))
        assert(inputs.size() == outputs.size(), "Invalid input size and output size.");
        assert(inputs.size() > 0, "Invalid input size, less than zero.");
    }

    void SigmoidLayer::backward(const std::vector<DataObject*>& inputs, std::vector<DataObject*>& outputs, std::vector<DataObject*>& input_diffs, std::vector<DataObject*>& output_diffs) {}
} // namespace DLHandsOn