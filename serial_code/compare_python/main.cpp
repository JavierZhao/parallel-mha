#include "multihead_attention.h"
#include <iostream>
#include <chrono>
#include "cnpy.h"

int main() {
    int num_heads = 2;
    int d_model = 4;
    int batch_size = 2;

    MultiHeadAttention mha(num_heads, d_model);

    // Load weights from the npz file
    cnpy::npz_t weights = cnpy::npz_load("mha_weights.npz");
    cnpy::NpyArray arr_Wq = weights["Wq"];
    cnpy::NpyArray arr_Wk = weights["Wk"];
    cnpy::NpyArray arr_Wv = weights["Wv"];
    cnpy::NpyArray arr_Wo = weights["Wo"];

    // Print dimensions of loaded weights
    std::cout << "Wq shape: " << arr_Wq.shape[0] << " x " << arr_Wq.shape[1] << std::endl;
    std::cout << "Wk shape: " << arr_Wk.shape[0] << " x " << arr_Wk.shape[1] << std::endl;
    std::cout << "Wv shape: " << arr_Wv.shape[0] << " x " << arr_Wv.shape[1] << std::endl;
    std::cout << "Wo shape: " << arr_Wo.shape[0] << " x " << arr_Wo.shape[1] << std::endl;

    // Convert to Eigen matrices
    MatrixXd Wq = Eigen::Map<MatrixXd>(arr_Wq.data<double>(), d_model, d_model);
    MatrixXd Wk = Eigen::Map<MatrixXd>(arr_Wk.data<double>(), d_model, d_model);
    MatrixXd Wv = Eigen::Map<MatrixXd>(arr_Wv.data<double>(), d_model, d_model);
    MatrixXd Wo = Eigen::Map<MatrixXd>(arr_Wo.data<double>(), d_model, d_model);

    std::cout << "Checking Wq\n";
    std::cout << Wq << std::endl;

    std::cout << "before set weight \n";

    mha.set_weights(Wq, Wk, Wv, Wo);

    std::cout << "after set weight \n";

    // Load input matrix x from the npy file
    cnpy::NpyArray arr_x = cnpy::npy_load("mha_input.npy");
    MatrixXd x = Eigen::Map<MatrixXd>(arr_x.data<double>(), batch_size, d_model);

    // Print dimensions of loaded input
    std::cout << "Input x shape: " << arr_x.shape[0] << " x " << arr_x.shape[1] << std::endl;
    std::cout << "Checking Input:\n";
    std::cout << x << std::endl;

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
