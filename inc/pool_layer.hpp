#ifndef POOL_LAYER_HPP
#define POOL_LAYER_HPP

#include "tensor.hpp"
#include "common.hpp"
#include "layer.hpp"
#include <cstddef>
#include <functional>

template <typename T>
class pool : public layer<T> {
public:
	pool(std::vector<int> kernel_shape,
		 std::vector<int> stride_shape,
		 std::vector<int> axes_to_pool,
		 pooling_type pool_type,
		 padding_type pad_type);

	pool(const tensor<T> input,
		 std::vector<int> kernel_shape,
		 std::vector<int> stride_shape,
		 std::vector<int> axes_to_pool,
		 pooling_type type,
		 padding_type pad_type);

	void initialize() override;

	void compute_output_shape() override;

	tensor<T> forward(tensor<T>& input) override;

	void backward(tensor<T>& gradient) override;

	void update_weights(T learning_rate) override;

	std::vector<int> get_output_shape() override;

	~pool();
	/* helper */
	std::vector<int> unravel_index(int flat_index, const std::vector<int>& shape);
	int ravel_index(const std::vector<int>& indices, const std::vector<int>& shape);
	std::vector<std::vector<int>> compute_window_starts();

	void compute_offset() {
		_kernel_indices.clear();
		int N = this->_input_shape.size();
		std::vector<int> current_index(N);
		std::function<void(int)> offset = [&](int depth) {
			if (depth == N) {
				_kernel_indices.push_back(current_index);
				return;
			}
			auto axis = std::find(_axes_to_pool.begin(), _axes_to_pool.end(), depth);
			if (axis != _axes_to_pool.end()) {
				int axis_index = std::distance(_axes_to_pool.begin(), axis);
				for (int i = 0; i < _kernel_shape[axis_index]; i++) {
					current_index[depth] = i;
					offset(depth + 1);
				}
			}
			else {
				current_index[depth] = 0;
				offset(depth + 1);
			}
		};
		offset(0);
	}
	//std::pair<T, int> apply_pooling_window(tensor<T>& data, std::vector<int>& start_index);
private:
	std::vector<int> _kernel_shape;
	std::vector<int> _stride;
	std::vector<int> _axes_to_pool;
	pooling_type _type;
	padding_type _pad_type;
	tensor<int> _mask;
	std::vector<std::vector<int>> _kernel_indices;
	std::vector<std::vector<int>> _window_starts;
	/* cache input and output */
	tensor<T> _input;
	tensor<T> _output;
};

/* constructor */
template <typename T>
pool<T>::pool(std::vector<int> kernel_shape,
			  std::vector<int> stride_shape,
			  std::vector<int> axes_to_pool,
			  pooling_type pool_type,
			  padding_type pad_type) : 
			  _kernel_shape(kernel_shape),
			  _stride(stride_shape),
			  _axes_to_pool(axes_to_pool),
			  _type(pool_type),
			  _pad_type(pad_type) {

}
// 2
template <typename T>
pool<T>::pool(const tensor<T> input,
			  std::vector<int> kernel_shape,
			  std::vector<int> stride_shape,
			  std::vector<int> axes_to_pool,
			  pooling_type pool_type,
			  padding_type pad_type) :
			  _input(input),
			  _kernel_shape(kernel_shape),
			  _stride(stride_shape),
			  _axes_to_pool(axes_to_pool),
			  _type(pool_type),
			  _pad_type(pad_type) {

}

template <typename T>
std::vector<int> pool<T>::get_output_shape() {
	return _output_shape;
}

template <typename T>
void pool<T>::compute_output_shape() {
	int N = this->_input_shape.size();
	this->_output_shape = this->_input_shape;

	for (int i = 0; i < N; i++) {
		auto axis = std::find(_axes_to_pool.begin(), _axes_to_pool.end(), i);
		if (axis != _axes_to_pool.end()) {
			int axis_index = std::distance(_axes_to_pool.begin(), axis);
			if (_pad_type == padding_type::SAME) {
				std::cout << "SMAE " << '\n';
				this->_output_shape[i] = static_cast<int>(std::ceil(this->_input_shape[i] / static_cast<float>(_stride[axis_index])));
			}
			else if (_pad_type == padding_type::VALID) {
				std::cout << "VALID " << '\n';
				this->_output_shape[i] = static_cast<int>(
				std::floor((this->_input_shape[i] - _kernel_shape[axis_index]) / static_cast<float>(_stride[axis_index])) + 1);
			}
			else if (_pad_type == padding_type::FULL) {
				std::cout << "FULL " << '\n';
				this->_output_shape[i] = static_cast<int>(std::ceil((this->_input_shape[i] + _kernel_shape[axis_index] - 1) / static_cast<float>(_stride[axis_index])));
			}

		}
	}
	_mask = tensor<int>(this->_output_shape);
	std::cout << "mask size: " << _mask.size();
}

/* helper */
template <typename T>
std::vector<int> pool<T>::unravel_index(int flat_index, const std::vector<int>& shape) {
    std::vector<int> indices(shape.size());
    std::vector<int> stride = tensor<T>::compute_stride(shape);
    for (int i = 0; i < shape.size(); i++) {
        indices[i] = flat_index / stride[i];
        flat_index = flat_index % stride[i];
    }
    return indices;
}
template <typename T>
int pool<T>::ravel_index(const std::vector<int>& indices, const std::vector<int>& shape) {
    std::vector<int> stride = tensor<T>::compute_stride(shape);
    int index = 0;
    for (int i = 0; i < shape.size(); i++) {
        index += indices[i] * stride[i];
    }
    return index;
}

template <typename T>
tensor<T> pool<T>::forward(tensor<T>& input) {
	_input = input;
	int kernel_size = std::accumulate(_kernel_shape.begin(), _kernel_shape.end(), static_cast<T>(1), std::multiplies<T>());
	int N = this->_input_shape.size();
	// if input size is similer then below code should run once and save it
	std::vector<std::vector<int>> window_starts = compute_window_starts();
	std::cout << "compute_window_starts : fine" << '\n';
	tensor<T> output(this->_output_shape);
	_mask.reshape(this->_output_shape);
	compute_offset();
	int out_index = 0;
	int temp_index = -1;
	for (auto& start : window_starts) {
		T value = 0;

		switch (_type) {
		case pooling_type::AVG:
			value = 0;
			break;
		case pooling_type::MAX:
			value = std::numeric_limits<T>::lowest();
			break;
		case pooling_type::MIN:
			value = std::numeric_limits<T>::max();
			break;
		}

		for (auto& offset : _kernel_indices) {
			std::vector<int> index(N);
			bool valid = true;
			for (int i = 0; i < N; ++i) {
				index[i] = start[i] + offset[i];
				if (index[i] < 0 || index[i] >= this->_input_shape[i]) {
					valid = false;
					break;
				}
			}
			if (valid) {
				int flat = ravel_index(index, this->_input_shape);
				T temp_value = input[flat];
				if (_type == pooling_type::MAX) {
					if (temp_value > value) {
						value = temp_value;
						temp_index = flat;
					}
				}
				else if (_type == pooling_type::MIN) {
					if (temp_value < value) {
						value = temp_value;
						temp_index = flat;
					}
				}
				else {
					value += temp_value;
				}
			}
		}
		if (_type == pooling_type::MAX || _type == pooling_type::MIN) {
			output[out_index] = value;
			if (temp_index >= 0) {
				_mask[out_index] = temp_index;
			}
		}
		else {
			output[out_index] = value / kernel_size;
		}
		out_index++;
	}
	std::cout << "mask" << '\n';
	for (int i = 0; i < _mask.size(); i++) {
		std::cout << _mask[i] << ' ';
	}
	_output = output;
	return output;
}

template <typename T>
std::vector<std::vector<int>> pool<T>::compute_window_starts() {
	int N = this->_input_shape.size();
	std::vector<std::vector<int>> index_ranges(N);

	for(int i = 0; i < N; i++) {
		auto it = std::find(_axes_to_pool.begin(), _axes_to_pool.end(), i);
		if (it != _axes_to_pool.end()) {
			int axis_index = std::distance(_axes_to_pool.begin(), it);
			int input_dim = this->_input_shape[i];
			int kernel = _kernel_shape[axis_index];
			int stride = _stride[axis_index];
			int pad_before = 0;
			int effective_input = input_dim;

			if (_pad_type == padding_type::SAME) {
				int out_dim = static_cast<int>(std::ceil(input_dim / static_cast<float>(stride)));
				int pad_total = std::max(static_cast<int>((out_dim - 1) * stride + kernel - input_dim), 0);
				pad_before = pad_total / 2;
				effective_input = input_dim + pad_total;
			}
			else if (_pad_type == padding_type::FULL) {
				std::cout << "else if FULL" << '\n';
				// For FULL, the effective input size is increased by (kernel - 1)
				pad_before = kernel - 1;
				std::cout << "pad_before " << pad_before << '\n';
				effective_input = input_dim + (kernel - 1) * 2;
				std::cout << "effective_input " << effective_input << '\n';
			}
			for (int j = -pad_before; j <= effective_input - kernel; j += stride) {
				index_ranges[i].push_back(j);
			}
		}
		else {
			index_ranges[i].push_back(0);
		}
	}
	std::vector<std::vector<int>> window_starts;
	std::vector<int> current(N, 0);
	std::function<void(int, std::vector<int>&)> cartesian_product = [&](int depth, std::vector<int>&current) {
		if (depth == N) {
			window_starts.push_back(current);
			return;
		}
		for (int value : index_ranges[depth]) {
			current[depth] = value;
			cartesian_product(depth + 1, current);
		}
	};
	cartesian_product(0, current);
	_window_starts = window_starts;
	return window_starts;
}
template <typename T>
void pool<T>::initialize() {}

template <typename T>
void pool<T>::backward(tensor<T>& gradient) {
	// check out size == gradient
		// reinitialize input to all 0

	tensor<T> input_gradient(this->_input_shape);

	_input.assign(static_cast<T>(0));
	for (int i = 0; i < _input.size(); i++) {
		std::cout << _input[i] << ' ';
	}
	if (_type == pooling_type::MAX || _type == pooling_type::MIN) {
		for (int i = 0; i < _mask.size(); i++) {
			_input[_mask[i]] = _output[i];
		}
	}
	else if (_type == pooling_type::AVG) {
		int kernel_size = 1;
		for (int i : _kernel_shape) {
			kernel_size *= i;
		}
		int out_index = 0;
		int N = this->_input_shape.size();

		for (auto& start : _window_starts) {			
			for (auto& offset : _kernel_indices) {
				std::vector<int> index(N, 0);
				bool valid = true;
				for (int i = 0; i < N; ++i) {

					index[i] = start[i] + offset[i];
					if (index[i] < 0 || index[i] >= N) {
						valid = false;
						break;
					}
				}
				if (valid) {
					int flat = ravel_index(index, this->_input_shape);
					_input[flat] += gradient[out_index] / static_cast<T>(kernel_size);
				}
			}
			out_index++;
		}
	}
	std::cout << "input shape" << _input.size() << '\n';
	for (int i = 0; i < _input.size(); i++) {
		std::cout << _input[i] << ' ';
	}
}

template <typename T>
void pool<T>::update_weights(T learning_rate) {
	// no learnable parameter
}
template <typename T>
pool<T>::~pool() {}

#endif // POOL_LAYER_HPP