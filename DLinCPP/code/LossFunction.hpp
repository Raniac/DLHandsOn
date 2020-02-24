#pragma once
#include "General.hpp"
#include "Utilities.hpp"
#include "DataObject.hpp"
#include <vector>

namespace DLHandsOn {
    class LossFunction {
    public:
        virtual float computeLoss(const std::vector<DataObject*>& ground_truth, const std::vector<DataObject*>& predictions) = 0;
        virtual std::vector<DataObject*> computeGrads(const std::vector<DataObject*>& ground_truth, const std::vector<DataObject*>& predictions) = 0;
    };

    class MSELoss : public LossFunction {
    public:
        virtual float computeLoss(const std::vector<DataObject*>& ground_truth, const std::vector<DataObject*>& predictions);
        virtual std::vector<DataObject*> computeGrads(const std::vector<DataObject*>& ground_truth, const std::vector<DataObject*>& predictions);
    };

    float MSELoss::computeLoss(const std::vector<DataObject*>& ground_truth, const std::vector<DataObject*>& predictions) {
        // compute mse loss
        // loss = 1/N * sum((y'i - yi) * (y'i - yi)), i belongs to [0, N)
        assert(ground_truth.size() == predictions.size(), "Invalid size of ground truth and predictions.");

        int num = 0;
        float total_loss = 0.0f;
        for (size_t i = 0; i < ground_truth.size(); i++) {
            assert(ground_truth[i] && predictions[i], "NULL predictions or ground truth.");
            const DataObject& gdtr = *ground_truth[i];
            const DataObject& pred = *predictions[i];
            assert(gdtr.getShape() == pred.getShape(), "Invalid shape of ground truth and predictions.");

            const DataType& gdtr_data = gdtr.getData();
            const DataType& pred_data = pred.getData();

            for (size_t j = 0; j < gdtr.getShape().getTotal(); j++) {
                num++;
                total_loss += (pred_data[j] - gdtr_data[j]) * (pred_data[j] - gdtr_data[j]);
            }
        }
        assert(num > 0, "Number of subject should be greater than zero.");
        return total_loss / num;
    }

    std::vector<DataObject*> MSELoss::computeGrads(const std::vector<DataObject*>& ground_truth, const std::vector<DataObject*>& predictions) {
        // compute gradients
        // grad = dloss/dy = 2.0f * (y' - y)
        assert(ground_truth.size() == predictions.size(), "Invalid size of ground truth and predictions.");

        std::vector<DataObject*> grads;
        for (size_t i = 0; i < ground_truth.size(); i++) {
            assert(ground_truth[i] && predictions[i], "NULL predictions or ground truth.");
            const DataObject& gdtr = *ground_truth[i];
            const DataObject& pred = *predictions[i];
            assert(gdtr.getShape() == pred.getShape(), "Invalid shape of ground truth and predictions.");

            const DataType& gdtr_data = gdtr.getData();
            const DataType& pred_data = pred.getData();

            DataObject* grad = new DataObject(pred.getShape());
            DataType& grad_data = grad->getData();

            for (size_t j = 0; j < gdtr.getShape().getTotal(); j++) {
                grad_data[j] = 2.0 * (pred_data[j] - gdtr_data[j]);
            }
            grads.push_back(grad);
        }
        return grads;
    }
} // namespace DLHandsOn