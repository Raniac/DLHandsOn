#pragma once
#include "Layer.hpp"
#include "Utilities.hpp"
#include "Optimizer.hpp"

namespace DLHandsOn {
    class DenseLayer : public Layer {
    public:
        DenseLayer();
        virtual ~DenseLayer();
    public:
        // setup layer
        void setup(const int input_size,
                   const int output_size,
                   const bool with_bias);

        const DataObject& getWeights() const;
        const DataObject& getBias() const;
        
        virtual std::vector<DataObject*> getAllWeights();
        virtual std::vector<DataObject*> getAllGrads();

        // TODO: add updateWeights function
        virtual void updateWeights(Optimizer* optimizer, int alpha);

        virtual void forward(const std::vector<DataObject*>& inputs,
                             std::vector<DataObject*>& outputs);
        virtual void backward(const std::vector<DataObject*>& inputs,
                              std::vector<DataObject*>& outputs,
                              std::vector<DataObject*>& input_diffs,
                              std::vector<DataObject*>& output_diffs);
    
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
        this->setPhase(Phase::Train);
        DLHandsOn::assert(this->getPhase() == Phase::Train, "Not Training!");
        if (this->getPhase() == Phase::Train) {
            weights_grad.reshape(weights.getShape());
        }

        if (with_bias) {
            bias.reshape(Shape({ output_size }));
            if (this->getPhase() == Phase::Train) { bias_grad.reshape(bias.getShape()); }
        }
        else bias.clear();

        // initiate weights
        uniformFiller(weights.getData(), -1.0f, 1.0f);

        // initiate bias
        constantFiller(bias.getData(), 0.0f);

        // initiate grads
        if (this->getPhase() == Phase::Train) {
            constantFiller(weights_grad.getData(), 0.0f);
            constantFiller(bias_grad.getData(), 0.0f);
        }
    }

    const DataObject& DenseLayer::getWeights() const { return weights; }

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
        return all_grads;
    }

    void DenseLayer::updateWeights(Optimizer* optimizer, int alpha) {
        optimizer->gradientDescent(weights, weights_grad, alpha);
        optimizer->gradientDescent(bias, bias_grad, alpha);
    }

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
    // let total loss, since
    // dy/dx = w, dy/dw = x, dy/db = 1
    // then
    // dloss/dx = dloss/dy * dy/dx = dloss/dy * w
    // dloss/dw = dloss/dy * dy/dw = dloss/dy * x
    // dloss/db = dloss/dy * dy/db = dloss/dy * 1
    void DenseLayer::backward(const std::vector<DataObject*>& inputs, std::vector<DataObject*>& outputs, std::vector<DataObject*>& input_diffs, std::vector<DataObject*>& output_diffs) {
        // assertion
        assert(inputs.size() == outputs.size(), "Invalid inputs and outputs size.");
        assert(input_diffs.size() == inputs.size(), "Invalid inputs and diffs size.");
        assert(output_diffs.size() == outputs.size(), "Invalid diffs and outputs size.");
        assert(inputs.size() > 0, "Invalid inputs size.");

        // for each input do
        for (size_t i = 0; i < inputs.size(); i++) {
            const DataObject& input = *inputs[i];
            DataObject& input_diff = *input_diffs[i];
            const DataObject& output_diff = *output_diffs[i];
            const DataObject& output = *outputs[i];

            const Shape& input_shape = input.getShape();
            const Shape& output_shape = output.getShape();
            const Shape& weights_shape = weights.getShape();
            const Shape& bias_shape = bias.getShape();

            // TODO: figure out what diff is and what it does with grad
            const Shape& weights_grad_shape = weights_grad.getShape();
            const Shape& bias_grad_shape = bias_grad.getShape();
            const Shape& input_diff_shape = input_diff.getShape();
            const Shape& output_diff_shape = output_diff.getShape();

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
                input_diff_shape == input_shape &&
                output_diff_shape == output_shape &&
                (bias.empty() ? true : output_size == bias_shape.getSize(0)),
                "Invalid dimensions.");

            // define results
            const DataType& input_data = input.getData();
            DataType& input_diff_data = input_diff.getData();
            const DataType& output_diff_data = output_diff.getData();
            const DataType& output_data = output.getData();
            const DataType& weights_data = weights.getData();
            const DataType& bias_data = bias.getData();
            DataType& weights_grad_data = weights_grad.getData();
            DataType& bias_grad_data = bias_grad.getData();
            
            const int batch_size = input_shape.getSize(0);
            // dy/dx = w: used for back propagation
            // dy: output_diff
            // dx: input_diff
            for (size_t j = 0; j < batch_size; j++) {
                for (size_t k = 0; k < input_size; k++) {
                    for (size_t l = 0; l < output_size; l++) {
                        // dloss/dx = dloss/dy * dy/dx = dloss/dy * w
                        input_diff_data[j * input_size + k] += weights_data[k * output_size + l] * output_diff_data[j * output_size + l];
                    }
                }
            }

            // dy/dw = x: used for weights updating
            weights_grad.fillValue(0.0f);
            for (size_t j = 0; j < batch_size; j++) {
                for (size_t k = 0; k < input_size; k++) {
                    for (size_t l = 0; l < output_size; l++) {
                        // dloss/dw = dloss/dy * dy/dw = dloss/dy * x
                        weights_grad_data[k * output_size + l] += input_data[j * input_size + k] * output_diff_data[j * output_size + l];
                    }
                }
            }
            for (size_t m = 0; m < weights_grad.getShape().getTotal(); m++) {
                weights_grad_data[m] /= batch_size;
            }

            if (!bias.empty()) {
                // dy/db = 1: used for bias updating
                for (size_t j = 0; j < batch_size; j++) {
                    for (size_t k = 0; k < input_size; k++) {
                        for (size_t l = 0; l < output_size; l++) {
                            // dloss/db = dloss/dy * dy/db = dloss/dy * 1
                            bias_grad_data[j * output_size + l] += 1.0f * output_diff_data[j * output_size + l];
                        }
                    }
                }
                for (size_t m = 0; m < bias_grad.getShape().getTotal(); m++) {
                    bias_grad_data[m] /= batch_size;
                }
            }
        }
    }
} // namespace DLHandsOn