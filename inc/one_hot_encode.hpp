#ifndef ONE_HOT_ENCODE
#define ONE_HOT_ENCODE

#include "tensor.hpp"
#include <fstream>
#include <sstream>
template<typename T>
void one_hot_encode(std::string path, tensor<T>& data, tensor<T>& tags, int rows, int cols, int depths) {
    std::ifstream file(path);
    if (!file.is_open()) {
        throw std::runtime_error("could not open CSV file: " + path);
    }

    std::unique_ptr<T[]> data_pointer = std::make_unique<T[]>(rows * cols * depths);
    std::unique_ptr<T[]> tags_pointer = std::make_unique<T[]>(10 * depths);
    std::string line;
    int depth = 0;
    while (std::getline(file, line) && depth < depths) {
        std::stringstream stream(line);
        std::string token;

        if (std::getline(stream, token, ',')) {
            throw std::runtime_error("missing data");
        }

        int tag = std::stoi(token);
        T* current_tag = tags_pointer.get() + (depth * 10);
        /* tag */
        for (int i = 0; i < 10; i++) {
            current_tag[i] = (i == static_cast<int>(tag)) ? static_cast<T>(1) : static_cast<T>(0);
        }
        /* data */
        T* current_data = data_pointer.get() + (depth * rows * cols);
        int dots = 0;
        while (std::getline(stream, token, ',') && dots < rows * cols) {
            current_data[dots] = static_cast<T>(std::stod(token));
            dots++;
        }
        depth++;
    }
    file.close();
    data._data = std::move(data_pointer);
    tags._data = std::move(tags_pointer);
}
#endif // ONE_HOT_ENCODE


//further optimization by +10 instead stride position calculation (lots of multiplication)