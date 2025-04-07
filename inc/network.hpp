#ifndef NETWORK_HPP
#define NETWORK_HPP

#include "tensor.hpp"
#include "layer.hpp"
#include <vector>
template <typename T>
class network {
public:
	network(std::vector<std::shared_ptr<layer<T>>> model);
	void train(int epoch);
	void output(std::string path);
	// serialize
	void serialize();
	// deserialize
	void deserialize();
private:
	std::vector<std::shared_ptr<layer<T>>> _model;
};
#endif // NETWORK_HPP

template <typename T>
network<T>::network(std::vector<std::shared_ptr<layer<T>>> model) : _model(model) {
	for (int i = 1; i < _model.size(); i++) {
		if (!_model[i]->has_input_shape()) {
			_model[i]->set_input_shape(_model[i - 1]->get_output_shape());
			_model[i]->compute_output_shape();
		}
	}
}
template <typename T>
void network<T>::train(int epoch) {
	// forward propagation pass -> compute error/gradients -> backward propagation pass
	while (epoch >= 0) {

		epoch--;
	}
}

template <typename T>
void network<T>::output(std::string path) {

}

template <typename T>
void network<T>::serialize() {

}
template <typename T>
void network<T>::deserialize() {

}

// initialize shape of hidden layer
// if next layer output_shape is not defined