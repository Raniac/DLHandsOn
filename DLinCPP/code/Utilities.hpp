#pragma once
#include <iostream>
#include <random>

namespace DLHandsOn {
    void assert(const bool condition, const char* message, ...) {
        if (!condition) {
            std::cout << message << std::endl;
        }
    }

    void constantFiller(DataType& data, const float val)
	{
		std::fill(data.begin(), data.end(), val);
	}

	void gaussianFiller(DataType& data, const float mean, const float std)
	{
		std::default_random_engine generator(std::random_device{}());
		std::normal_distribution<float> distribution(mean, std);
		for (size_t i = 0; i < data.size(); i++)
		{
			data[i] = distribution(generator);
		}
	}

	void uniformFiller(DataType& data, const float min_val, const float max_val)
	{
		std::default_random_engine generator(std::random_device{}());
		std::uniform_real_distribution<float> distribution(min_val, max_val);
		for (size_t i = 0; i < data.size(); i++)
		{
			data[i] = distribution(generator);
		}
	}
} // DLHandsOn