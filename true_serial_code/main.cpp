#include "multihead_attention.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <chrono>
#include <vector>
#include <string>

// Helper function to load a matrix from a CSV file
Matrix load_csv(const std::string& file_path, int rows, int cols) {
    Matrix matrix(rows, cols);
    std::ifstream file(file_path);
    std::string line;
    int row = 0;
    while (getline(file, line)) {
        std::stringstream stream(line);
        std::string cell;
        int col = 0;
        while (getline(stream, cell, ',')) {
            matrix.data[row][col++] = std::stod(cell);
        }
        ++row;
    }
    file.close();
    return matrix;
}

int main() {
    int num_heads = 2;
    int d_model = 4;
    int seq_length = 2;

    MultiHeadAttention mha(num_heads, d_model);

    // Load weights, biases, input and expected output from CSV files
    Matrix Wq = load_csv("input/Wq.csv", d_model, d_model);
    Matrix Wk = load_csv("input/Wk.csv", d_model, d_model);
    Matrix Wv = load_csv("input/Wv.csv", d_model, d_model);
    Matrix Wo = load_csv("input/Wo.csv", d_model, d_model);

    Matrix bq = load_csv("input/bq.csv", 1, d_model);
    Matrix bk = load_csv("input/bk.csv", 1, d_model);
    Matrix bv = load_csv("input/bv.csv", 1, d_model);
    Matrix bo = load_csv("input/bo.csv", 1, d_model);

    Matrix x = load_csv("input/mha_input.csv", seq_length, d_model);
    Matrix expected_output = load_csv("input/mha_output.csv", seq_length, d_model);

    // print weights
    std::cout << "Wq:\n" << Wq;
    std::cout << "Wk:\n" << Wk;
    std::cout << "Wv:\n" << Wv;
    std::cout << "Wo:\n" << Wo;

    // Set weights
    mha.set_weights(Wq, Wk, Wv, Wo);

    // print bias
    std::cout << "bq:\n" << bq << std::endl;
    std::cout << "bk:\n" << bk << std::endl;
    std::cout << "bv:\n" << bv << std::endl;
    std::cout << "bo:\n" << bo << std::endl;

    // Set bias
    mha.set_biases(bq, bk, bv, bo); 

    // Print input matrix
    std::cout << "Input:\n" << x << std::endl;

    auto start = std::chrono::high_resolution_clock::now();

    // Compute output
    Matrix output = mha.compute(x);

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;

    // Print results
    std::cout << "Output:\n" << output << std::endl;
    std::cout << "Execution time: " << duration.count() << " seconds\n";

    // Print expected output
    std::cout << "Expected output:\n" << expected_output << std::endl;

    // Compare the outputs, element-wise within a tolerance
    double tolerance = 1e-5;
    for (int i = 0; i < seq_length; ++i) {
        for (int j = 0; j < d_model; ++j) {
            if (std::abs(output.data[i][j] - expected_output.data[i][j]) > tolerance) {
                std::cerr << "Mismatch at position (" << i << ", " << j << "): "
                          << output.data[i][j] << " vs " << expected_output.data[i][j] << std::endl;
                return 1;
            }
        }
    }
    std::cout << "Output matches expected output\n";


    return 0;
}
