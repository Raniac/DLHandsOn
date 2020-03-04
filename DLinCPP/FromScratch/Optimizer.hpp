#pragma once
#include "General.hpp"
#include "Utilities.hpp"
#include "DataObject.hpp"
#include <vector>

namespace DLHandsOn {
    class Optimizer {
    public:
        void gradientDescent(DataObject& weights,
                            DataObject& grads,
                            int alpha);
    };

    void Optimizer::gradientDescent(DataObject& weights, DataObject& weights_grad, int alpha) {
        // Gradient Descent
        // w = w - a * grad
        const Shape& weights_shape = weights.getShape();
        const Shape& weights_grad_shape = weights_grad.getShape();
        assert(weights_shape == weights_grad_shape, "Invalid weights and grads shape.");

        const DataType& weights_grad_data = weights_grad.getData();
        DataType& weights_data = weights.getData();

        if (weights_shape.getDims() == 2) { // update weights
            for (size_t i = 0; i < weights_shape.getSize(0); i++) {
                for (size_t j = 0; j < weights_shape.getSize(1); j++) {
                    weights_data[i * weights_shape.getSize(1) + j] -= alpha * weights_grad_data[i * weights_shape.getSize(1) + j];
                }
            }
        }
        else if (weights_shape.getDims() == 1) { // update bias
            for (size_t i = 0; i < weights_shape.getSize(0); i++) {
                    weights_data[i] -= alpha * weights_grad_data[i];
            }
        }
    }
} // namespace DLHandsOn