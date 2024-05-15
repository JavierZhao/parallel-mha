// main.cpp
#include "multihead_attention.h"
#include <iostream>
#include <chrono>

int main() {
    int num_heads = 4;
    int d_model = 128;
    int batch_size = 2;

    MultiHeadAttention mha(num_heads, d_model);

    MatrixXd x = MatrixXd::Random(batch_size, d_model);

    // Start the timer
    auto start = std::chrono::high_resolution_clock::now();

    MatrixXd output = mha.compute(x);

    // Stop the timer
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;

    std::cout << "Output:\n" << output << std::endl;
    std::cout << "Execution time: " << duration.count() << " seconds\n";
    std::cout << "Model dimensions: num_heads = " << num_heads << ", d_model = " << d_model << std::endl;
    std::cout << "Matrix dimensions: batch_size = " << batch_size << std::endl;

    return 0;
}
