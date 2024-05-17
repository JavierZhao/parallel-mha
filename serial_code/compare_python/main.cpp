#include "multihead_attention.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <chrono>
#include <vector>
#include <string>
#include <Eigen/Dense>

using Eigen::MatrixXd;

// Helper function to load a matrix from a CSV file
MatrixXd load_csv(const std::string& file_path, int rows, int cols) {
    MatrixXd matrix(rows, cols);
    std::ifstream file(file_path);
    std::string line;
    int row = 0;
    while (getline(file, line)) {
        std::stringstream stream(line);
        std::string cell;
        int col = 0;
        while (getline(stream, cell, ',')) {
            matrix(row, col++) = std::stod(cell);
        }
        ++row;
    }
    file.close();
    return matrix;
}


int main() {
    int num_heads = 2;
    int d_model = 4;
    int batch_size = 2;

    MultiHeadAttention mha(num_heads, d_model);

    // Load weights from CSV files
    MatrixXd Wq = load_csv("Wq.csv", d_model, d_model);
    MatrixXd Wk = load_csv("Wk.csv", d_model, d_model);
    MatrixXd Wv = load_csv("Wv.csv", d_model, d_model);
    MatrixXd Wo = load_csv("Wo.csv", d_model, d_model);

    // Set weights
    mha.set_weights(Wq, Wk, Wv, Wo);

    // Load input matrix x from CSV file
    MatrixXd x = load_csv("mha_input.csv", batch_size, d_model);

    auto start = std::chrono::high_resolution_clock::now();

    // Compute output
    MatrixXd output = mha.compute(x);

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;

    // Print results
    std::cout << "Output:\n" << output << std::endl;
    std::cout << "Execution time: " << duration.count() << " seconds\n";


    // Load expected output from CSV file
    MatrixXd expected_output = load_csv("mha_output.csv", batch_size, d_model);

    // Compare the outputs
    if (output.isApprox(expected_output, 1e-6)) {
        std::cout << "The outputs match!" << std::endl;
    } else {
        std::cout << "The outputs do not match." << std::endl;
    }

    return 0;
}
