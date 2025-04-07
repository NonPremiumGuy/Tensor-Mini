#ifndef INITIALIZER_HPP
#define INITIALIZER_HPP

#include "tensor.hpp"
#include <random>
#include <cmath>

template <typename T>
class initializer {
public:
	virtual ~initializer() {};
	//NOTE: fan_in = number of input coonections to a neuron
	//fan_out = number of output connection from a neuron
	virtual void init(tensor<T>& W, int fan_in, int fan_out) = 0;
private:
protected:
};

/* He Initializer */
template <typename T>
class He_init : public initializer<T> {
public:
	void init(tensor<T>& W, int fan_in, int fan_out) override;
};

template <typename T>
void He_init<T>::init(tensor<T>& W, int fan_in, int fan_out) {
	std::random_device RD;
	std::mt19937 gen(RD());
	T std_dev = std::sqrt(2.0 / fan_in);
	std::normal_distribution<T> dist(0.0, std_dev);

	for (T& weight : W) {
		weight = dist(gen);
	}
}

#endif // INITIALIZER_HPP