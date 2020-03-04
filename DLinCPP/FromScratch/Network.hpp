#pragma once
#include "General.hpp"
#include "DataObject.hpp"
#include "Layer.hpp"
#include "Optimizer.hpp"
#include "LossFunction.hpp"

namespace DLHandsOn {
    class Network {
    public:
        Network& addLayer(Layer* layer);

        void setLoss(LossFunction* loss_function);
        float getLoss(const std::vector<DataObject*>& ground_truth, const std::vector<DataObject*>& predictions);
        std::vector<DataObject*> getGrads(const std::vector<DataObject*>& ground_truth, const std::vector<DataObject*>& predictions);
        
        void forward(const std::vector<DataObject*>& inputs, std::vector<DataObject*>& predictions);
        void backward(const std::vector<DataObject*>& inputs, std::vector<DataObject*>& predictions, const std::vector<DataObject*>& grads, Optimizer* optimizer);
        void updateWeights();

        bool saveModel(const std::string& model_path) const;
    private:
        std::vector<Layer*> layers;
        LossFunction* loss_function = nullptr;
    };

    Network& Network::addLayer(Layer* layer) {
        assert(layer != nullptr, "Null layer.");
        layers.push_back(layer);
        return *this;
    }

    void Network::setLoss(LossFunction* loss_function) {
        assert(loss_function != nullptr, "Null layer.");
        this->loss_function = loss_function;
    }

    // TODO: use loss function to get loss
    float Network::getLoss(const std::vector<DataObject*>& ground_truth, const std::vector<DataObject*>& predictions) {
        return loss_function->computeLoss(ground_truth, predictions);
    }

    // TODO: use loss function to get grads
    std::vector<DataObject*> Network::getGrads(const std::vector<DataObject*>& ground_truth, const std::vector<DataObject*>& predictions) {
        return loss_function->computeGrads(ground_truth, predictions);
    }

    // TODO: set input size and output size for each layer separately
    void Network::forward(const std::vector<DataObject*>& inputs, std::vector<DataObject*>& predictions) {
        std::vector<DataObject*> temp_inputs = inputs;
        std::vector<DataObject*> temp_outputs = predictions;
        for (auto layer : layers) {
            layer->forward(temp_inputs, temp_outputs);
            temp_inputs = temp_outputs;
        }
        predictions = temp_outputs;
    }

    // TODO: call backward function of each layer
    void Network::backward(const std::vector<DataObject*>& inputs, std::vector<DataObject*>& predictions, const std::vector<DataObject*>& grads, Optimizer* optimizer) {
        std::vector<DataObject*> input_diffs = inputs;
        std::vector<DataObject*> output_diffs = grads;
        for (size_t i = layers.size() - 1; i > 0; i--) {
            layers[i]->backward(inputs, predictions, input_diffs, output_diffs);
            output_diffs = input_diffs;
        }
    }
} // namespace DLHandsOn