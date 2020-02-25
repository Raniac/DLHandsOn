#pragma once
#include "General.hpp"
#include "DataObject.hpp"
#include "Layer.hpp"
#include "LossFunction.hpp"

namespace DLHandsOn {
    class Network {
    public:
        Network& addLayer(Layer* layer);

        void setLoss(LossFunction* loss_function);
        float getLoss(const std::vector<DataObject*>& ground_truth, const std::vector<DataObject*>& predictions);

        void forward(const std::vector<DataObject*>& inputs, std::vector<DataObject*>& predictions);
        void backward(const std::vector<DataObject*>& inputs, const std::vector<DataObject*>& ground_truth);
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
    float Network::getLoss(const std::vector<DataObject*>& ground_truth, const std::vector<DataObject*>& predictions) {}

    // TODO: call forward function of each layer
    void Network::forward(const std::vector<DataObject*>& inputs, std::vector<DataObject*>& predictions) {}

    // TODO: call backward function of each layer
    void Network::backward(const std::vector<DataObject*>& inputs, const std::vector<DataObject*>& ground_truth) {}

    // TODO: update hidden layers' weights
    void Network::updateWeights() {}
} // namespace DLHandsOn