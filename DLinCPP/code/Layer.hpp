#pragma once
#include <vector>
#include "General.hpp"
#include "DataObject.hpp"

namespace DLHandsOn {
    class Layer {
    public:
        enum class Phase {
            Train,
            Test
        };
        void setPhase(const Phase phase) { this->phase = phase; }
        Phase getPhase() const { return this->phase; }

        // get all the weights of this layer
        virtual std::vector<DataObject*> getAllWeights() = 0;

        // get all the gradients of this layer
        virtual std::vector<DataObject*> getAllGrads() = 0;

        // forward function
        virtual void forward(const std::vector<DataObject*>& inputs,
                             std::vector<DataObject*>& outputs) = 0;

        // backward function
        virtual void backward(const std::vector<DataObject*>& inputs,
                              std::vector<DataObject*>& outputs,
                              std::vector<DataObject*>& prev_diffs,
                              std::vector<DataObject*>& next_diffs) = 0;

    private:
        Phase phase = Phase::Test;
    };
} // namespace DLHandsOn