#pragma once
#include "Layer.hpp"
#include "Utilities.hpp"

namespace DLHandsOn {
    class DenseLayer : public Layer {
    public:
        DenseLayer();
        virtual ~DenseLayer();
    public:
        // setup layer
        void setup(const int input_dim,
                   const int output_dim,
                   const bool with_bias);

        const DataObject& getWeight() const;
        const DataObject& getBias() const;
        
        virtual std::vector<DataObject*> getAllWeights();
        virtual std::vector<DataObject*> getAllGrads();

        virtual void forward(const std::vector<DataObject*>& inputs,
                             std::vector<DataObject*>& outputs);
        virtual void backward(const std::vector<DataObject*>& inputs,
                              std::vector<DataObject*>& outputs,
                              std::vector<DataObject*>& prev_diffs,
                              std::vector<DataObject*>& next_diffs);
    
    private:
        DataObject weights;
        DataObject weights_grad;
        DataObject bias;
        DataObject bias_grad;
    };

    DenseLayer::DenseLayer() {}

    DenseLayer::~DenseLayer() {}

    void DenseLayer::setup(const int input_size, const int output_size, const bool with_bias) {
        weights.reshape(Shape({ input_size, output_size }));
        if (getPhase() == Phase::Train) {
            weights_grad.reshape(weights.getShape());
        }

        if (with_bias) {
            bias.reshape(Shape({ output_size }));
            if (getPhase() == Phase::Test) bias_grad.reshape(bias.getShape());
        }
        else bias.clear();

        // initiate weights
        uniformFiller(weights.getData(), -1.0f, 1.0f);

        // initiate bias
        constantFiller(bias.getData(), 0.0f);

        // initiate grads
        if (getPhase() == Phase::Train) {
            constantFiller(weights_grad.getData(), 0.0f);
            constantFiller(bias_grad.getData(), 0.0f);
        }
    }

    const DataObject& DenseLayer::getWeight() const { return weights; }

    const DataObject& DenseLayer::getBias() const { return bias; }

    std::vector<DataObject*> DenseLayer::getAllWeights() {
        std::vector<DataObject*> all_weights;
        all_weights.push_back(&weights);
        all_weights.push_back(&bias);
        return all_weights;
    }
    std::vector<DataObject*> DenseLayer::getAllGrads() {
        std::vector<DataObject*> all_grads;
        all_grads.push_back(&weights_grad);
        all_grads.push_back(&bias_grad);
        return all_grads;}

    // forward inference
    // y = w * x + b
    void DenseLayer::forward(const std::vector<DataObject*>& inputs, std::vector<DataObject*>& outputs) {
        assert(inputs.size() == outputs.size(), "Invalid input size and output size");

        for (size_t i = 0; i < inputs.size(); i++) {
            const DataObject& input = *inputs[i];
            DataObject&  output = *outputs[i];

            const Shape& input_shape = input.getShape();
            const Shape& output_shape = output.getShape();
            const Shape& weights_shape = weights.getShape();
            const Shape& bias_shape = bias.getShape();

            const int input_size = input_shape.getSize(1);
            const int output_size = output_shape.getSize(1);

            // validate shapes
            // shape of input: (N, input_size)
            // shape of output: (N, output_size)
            // shape of weights: (input_size, output_size)
            // shape of bias: (output_size)
            assert(input_shape.getDims() == output_shape.getDims() &&
                input_shape.getSize(0) == output_shape.getSize(0) &&
                input_size == weights_shape.getSize(0) &&
                output_size == weights_shape.getSize(1),
                "Invalid dimensions.");

            const DataType& input_data = input.getData();
            DataType& output_data = output.getData();
            const DataType& weights_data = weights.getData();
            const DataType& bias_data = bias.getData();
            const int batch_size = input_shape.getSize(0);

            /*
            Suppose:
                X = [x1 x2 x3], W = [[w1 w2 w3] [w4 w5 w6]], B = [b1 b2], Y = [y1 y2]
            Then:
                Y = W X' + B
            Thus:
                y1 = x1*w1 + x2*w2 + x3*w3 + b1, y2 = x4*w4 + x5*w5 + x6*w6 + b2
            */
            for (size_t j = 0; j < batch_size; j++) {
                for (size_t k = 0; k < output_size; k++) {
                    float sum = 0.0f;
                    for (size_t l = 0; l < input_size; l++) {
                        sum += weights_data[l * output_size + k] * input_data[j * input_size + l];
                    }
                    if (!bias.empty()) sum += bias_data[k];
                    output_data[j * output_size + k] = sum;
                }
            }
        }
    }

    // backward propagation
    // dy/dx = w, dy/dw = x, dy/db = 1
    void DenseLayer::backward(const std::vector<DataObject*>& inputs, std::vector<DataObject*>& outputs, std::vector<DataObject*>& prev_diffs, std::vector<DataObject*>& next_diffs) {
        // assertion
        assert(inputs.size() == outputs.size(), "Invalid inputs and outputs size.");
        assert(prev_diffs.size() == inputs.size(), "Invalid inputs and diffs size.");
        assert(next_diffs.size() == outputs.size(), "Invalid diffs and outputs size.");
        assert(inputs.size() > 0, "Invalid inputs size.");

        // for each input do
        for (size_t i = 0; i < inputs.size(); i++) {
            const DataObject& input = *inputs[i];
            DataObject& prev_diff = *prev_diffs[i];
            const DataObject& next_diff = *next_diffs[i];
            const DataObject& output = *outputs[i];

            const Shape& input_shape = input.getShape();
            const Shape& output_shape = output.getShape();
            const Shape& weights_shape = weights.getShape();
            const Shape& bias_shape = bias.getShape();

            // TODO: figure out what diff is and what it does with grad
            const Shape& weights_grad_shape = weights_grad.getShape();
            const Shape& bias_grad_shape = bias_grad.getShape();
            const Shape& prev_diff_shape = prev_diff.getShape();
            const Shape& next_diff_shape = next_diff.getShape();

            const int input_size = input_shape.getSize(1);
            const int diff_size = input_size;
            const int output_size = output_shape.getSize(1);

            // validate shapes
            // shape of input: (N, input_size)
            // shape of output: (N, output_size)
            // shape of weights: (input_size, output_size)
            // shape of bias: (output_size)
            assert(input_shape.getDims() == output_shape.getDims() &&
                input_shape.getSize(0) == output_shape.getSize(0) &&
                input_size == weights_shape.getSize(0) &&
                output_size == weights_shape.getSize(1) &&
                weights_grad_shape == weights_shape &&
                bias_grad_shape == bias_shape &&
                prev_diff_shape == input_shape &&
                next_diff_shape == output_shape &&
                (bias.empty() ? true : output_size == bias_shape.getSize(0)),
                "Invalid dimensions.");
        }
    }
} // namespace DLHandsOn