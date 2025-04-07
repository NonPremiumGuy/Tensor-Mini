#ifndef COMMON_HPP
#define COMMON_HPP
enum class padding_type {
	VALID, // no padding so that output shape < input shape
	SAME, // 0 padding so that output shape = input shape
	FULL // 
};
enum class pooling_type {
	MAX,
	MIN,
	AVG
};
enum class activation_type {
	ReLU,
	Sigmoid,
	Leaky_ReLU,
	TanH,
	SoftMax
};
#endif // COMMON_HPP