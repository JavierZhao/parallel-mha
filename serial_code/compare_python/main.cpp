#include "multihead_attention.h"
#include <iostream>
#include <chrono>
#include "cnpy.h"

int main() {
    int num_heads = 4;
    int d_model = 128;
    int batch_size = 2;

    MultiHeadAttention mha(num_heads, d_model);

    // Load weights from the npz file
    cnpy::npz_t weights = cnpy::npz_load("mha_weights.npz");
    cnpy::NpyArray arr_Wq = weights["Wq"];
    cnpy::NpyArray arr_Wk = weights["Wk"];
    cnpy::NpyArray arr_Wv = weights["Wv"];
    cnpy::NpyArray arr_Wo = weights["Wo"];

    // Convert to Eigen matrices
    MatrixXd Wq = Eigen::Map<MatrixXd>(arr_Wq.data<double>(), d_model, d_model);
    MatrixXd Wk = Eigen::Map<MatrixXd>(arr_Wk.data<double>(), d_model, d_model);
    MatrixXd Wv = Eigen::Map<MatrixXd>(arr_Wv.data<double>(), d_model, d_model);
    MatrixXd Wo = Eigen::Map<MatrixXd>(arr_Wo.data<double>(), d_model, d_model);

    mha.set_weights(Wq, Wk, Wv, Wo);

    // Load input matrix x from the npy file
    cnpy::NpyArray arr_x = cnpy::npy_load("mha_input.npy");
    MatrixXd x = Eigen::Map<MatrixXd>(arr_x.data<double>(), batch_size, d_model);

    auto start = std::chrono::high_resolution_clock::now();

    MatrixXd output = mha.compute(x);

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;

    std::cout << "Output:\n" << output << std::endl;
    std::cout << "Execution time: " << duration.count() << " seconds\n";
    std::cout << "Model dimensions: num_heads = " << num_heads << ", d_model = " << d_model << std::endl;
    std::cout << "Matrix dimensions: batch_size = " << batch_size << std::endl;

    // Load expected output from the npy file
    cnpy::NpyArray arr_expected_output = cnpy::npy_load("mha_output.npy");
    MatrixXd expected_output = Eigen::Map<MatrixXd>(arr_expected_output.data<double>(), batch_size, d_model);

    std::cout << "Expected Output:\n" << expected_output << std::endl;

    // Compare the outputs
    if (output.isApprox(expected_output, 1e-6)) {
        std::cout << "The outputs match!" << std::endl;
    } else {
        std::cout << "The outputs do not match." << std::endl;
    }

    return 0;
}
