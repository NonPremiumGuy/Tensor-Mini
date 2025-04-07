#ifndef LAYER_HPP
#define LAYER_HPP

#include "tensor.hpp"
#include "initializer.hpp"

template <typename T>
class layer {
public:
	layer();
	std::vector<int> _input_shape;
	std::vector<int> _output_shape;
	/* NOTE: virtual function() = 0; <- pure virtual
	and any class that inherits (derived classes) 
	from this base class must implement it.
	*/
	virtual void initialize(initializer<T>& init) = 0;
	virtual void compute_output_shape() = 0;
	virtual tensor<T>forward(tensor<T>& input) = 0;
	virtual void backward(tensor<T>& gradient) = 0;
	virtual void update_weights(T learning_rate) = 0;
	virtual std::vector<int> get_output_shape();
	virtual void set_input_shape(std::vector<int>& input_shape);
	virtual bool has_input_shape();
	virtual ~layer();
};

template <typename T>
layer<T>::layer() {}

template <typename T>
std::vector<int> layer<T>::get_output_shape() {
	return _output_shape;
}

template <typename T>
void layer<T>::set_input_shape(std::vector<int>& input_shape) {
	_input_shape = input_shape;
	compute_output_shape();
}

template <typename T>
bool layer<T>::has_input_shape() {
	return !_input_shape.empty();
}

template <typename T>
layer<T>::~layer() = default;

#endif // LAYER_HPP