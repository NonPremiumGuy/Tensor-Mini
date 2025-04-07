#ifndef TENSOR_H
#define TENSOR_H

#include <vector>
#include <functional>
#include <stdexcept>
#include <numeric>
#include <memory>
#include <iostream>
#include <cstddef>

template <typename T>
class tensor {
public:
    //void print() {
    //    _shape
    //}
    /* CONSTRUCTOR */
    /* 1. */
    tensor();
    /* 2. */
    tensor(std::vector<int>& dimension);
    /* 3. */
    tensor(std::vector<int>& dimension, std::shared_ptr<T[]> data);
    /* 4 */
    tensor(std::shared_ptr <T[]> data, T* start, std::vector<int> shape, std::vector<int> stride, int offset);
    /* 5 */
    tensor(std::vector<int>& dimension, T value);
    /* copy constructor */
    tensor(const tensor<T>& other);
    /* assignment operator */
    tensor<T>& operator=(const tensor<T>& other);
    /* move constructor */
    tensor(tensor<T>&& other) = default;

    /* access operator */
    /* 1 */
    T& operator[](std::vector<int> indices);
    T& operator[](int index);
    /* destructor */
    ~tensor();

    /* reshape */
    void reshape(std::vector<int>& newDimension);

    /* slice */
    tensor<T> slice(int dimension, int start, int end);
    /* transpose */
    tensor transpose();
    /* stride */
    std::vector<int> stride() {
        return _stride;
    }
    /* size */
    int size();
    /* number of dimensions */
    int ndims();
    /* vector of -> number of elements in each dimension */
    std::vector<int> dims();
    /**
    * NOTE:
    * TODO:
    */
    void fill();
    void assign(T value) {
        for (int i = 0; i < _size; i++) {
            _data.get()[i] = value;
        }
    }
    tensor operator+(const tensor& other);
    tensor operator-(const tensor& other);
    tensor operator*(const tensor& other);
    tensor operator/(const tensor& other);

    tensor arithmetic(const tensor& other, std::function<T(T, T)> op);

protected:
    std::shared_ptr<T[]> _data;
    std::vector<int> _shape;
    std::vector<int> _stride;
    int _size;
    T* _start = NULL;
    int _offset = 0;
public:
    static std::vector<int> compute_stride(const std::vector<int>& dimension);
    std::vector<int> dims() const;
};

/* CONSTRUCTOR */
/* 1. */
template <typename T>
tensor<T>::tensor() : _data(), _shape(), _stride(), _size(0) {};
/* 2. */
template <typename T>
tensor<T>::tensor(std::vector<int>& shape) : _shape(shape) {
    _size = 1;
    for (int dim : shape) {
        _size *= dim;
    }
    _stride = compute_stride(shape);
    _data = std::shared_ptr<T[]>(new T[_size]);
    _start = _data.get();
}
/* 3. */
template <typename T>
tensor<T>::tensor(std::vector<int>& shape, std::shared_ptr<T[]> data) : _shape(shape), _data(std::move(data)) {
    _size = 1;
    for (int dim : shape) {
        _size *= dim;
    }
    _stride = compute_stride(shape);
    _start = _data.get();
}
/* 4 */
template <typename T>
tensor<T>::tensor(std::shared_ptr <T[]> data, T* start, std::vector<int> shape, std::vector<int> stride, int offset) : _data(std::move(data)), _start(start), _shape(std::move(shape)), _stride(std::move(stride)), _offset(offset) {
    _size = 1;
    for (int d : _shape) {
        _size *= d;
    }
}
/* 5 */
template <typename T>
tensor<T>::tensor(std::vector<int>& shape, T value) : _shape(shape) {
    _size = 1;
    for (int dim : shape) {
        _size *= dim;
    }
    _stride = compute_stride(shape);
    _data = std::shared_ptr<T[]>(new T[_size]);
    _start = _data.get();
    for (int i = 0; i < _size; i++) {
        this->_data.get()[i] = value;
    }
}

/* copy constructor */
template <typename T>
tensor<T>::tensor(const tensor<T>& other) {
    _size = other._size;
    _shape = other._shape;
    _stride = other._stride;
    _offset = other._offset;

    _data = other._data;
    _start = other._start;
}
/* assignment operator */
template <typename T>
tensor<T>& tensor<T>::operator=(const tensor<T>& other) {
    if (this == &other) {
        return *this;
    }
    _shape = other._shape;
    _stride = other._stride;
    _offset = other._offset;
    _size = other._size;
    _data = other._data;
    _start = other._start;
    return *this;
}

/* access operator */
/* 1 */
template <typename T>
T& tensor<T>::operator[](std::vector<int> indices) {
    if (_stride.size() != indices.size()) {
        throw std::out_of_range("index dimension mismatch");
    }
    int index = _offset;
    for (int i = 0; i < indices.size(); i++) {
        if (indices[i] >= _shape[i]) {
            throw std::out_of_range("index out of bounds");
        }
        index += indices[i] * _stride[i];
    }
    return _start[index];
}
/* 2 */
template <typename T>
T& tensor<T>::operator[](int index) {
    return _start[index];
}
/* destructor */
template <typename T>
tensor<T>::~tensor() {};

/* reshape */
template <typename T>
void tensor<T>::reshape(std::vector<int>& new_shape) {
    int new_size = 1;
    for (auto dim : new_shape) {
        new_size *= dim;
    }
    if (_size != new_size) {
        throw std::invalid_argument("new shape size mismatch");
    }
    _size = new_size;
    _shape = new_shape;
    _stride = compute_stride(new_shape);
}

/* slice */
template <typename T>
tensor<T> tensor<T>::slice(int dim, int start, int end) {
    if (start >= end || dim >= _shape.size() || end > _shape[dim]) {
        throw std::out_of_range("invalid slice range");
    }
    std::vector<int> new_shape = _shape;
    new_shape[dim] = end - start;

    int new_offset = _offset + start * _stride[dim];
    return tensor(_data, _start + new_offset, new_shape, _stride, new_offset);
}
template <typename T>
tensor<T> tensor<T>::transpose() {
    tensor<T> transposed;
    return transposed;
}
/* size */
template <typename T>
int tensor<T>::size() {
    return _size;
}
/* number of dimensions */
template <typename T>
int tensor<T>::ndims() {
    return _shape.size();
}
/* vector of -> number of elements in each dimension */
template <typename T>
std::vector<int> tensor<T>::dims() {
    return _shape;
}

template <typename T>
tensor<T> tensor<T>::operator+(const tensor<T>& other) {
    return arithmetic(other, std::plus<T>());
}

template <typename T>
tensor<T> tensor<T>::operator-(const tensor<T>& other) {
    return arithmetic(other, std::minus<T>());

}

template <typename T>
tensor<T> tensor<T>::operator*(const tensor<T>& other) {
    return arithmetic(other, std::multiplies<T>());

}

template <typename T>
tensor<T> tensor<T>::operator/(const tensor<T>& other) {
    return arithmetic(other, [](T a, T b) {
        if (b == static_cast<T>(0)) {
            throw std::runtime_error("division by zero");
        }
        return a / b;
        });
}

template <typename T>
tensor<T> tensor<T>::arithmetic(const tensor<T>& other, std::function<T(T, T)> op) {
    if (this->_shape != other._shape) {
        throw std::invalid_argument("Tensor dimensions must match for arithmetic operations");
    }

    // Create a new tensor to hold the result
    tensor<T> result(this->_shape);
    int dataLength = this->_size;

    // Perform the operation element-wise
    for (int i = 0; i < dataLength; ++i) {
        result._data[i] = op(this->_data[i], other._data[i]);
    }

    return result;
}

/* private: */
template <typename T>
std::vector<int> tensor<T>::compute_stride(const std::vector<int>& shape) {
    std::vector<int> stride(shape.size());
    int skip = 1;
    for (int i = shape.size(); i-- > 0;) {
        stride[i] = skip;
        skip *= shape[i];
    }
    return stride;
}

/* const version */
template <typename T>
std::vector<int> tensor<T>::dims() const {
    return _shape;
}

#endif // TENSOR_H