#include "multihead_attention.h"
#include "matrix.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <chrono>
#include <vector>
#include <string>

#include "mpi.h"

// extern "C" {
//     #include "summa.h"  // Include the C header within extern "C" if it's not C++ ready
// }

// Helper function to load a matrix from a CSV file
// Matrix load_csv(const std::string& file_path, int rows, int cols) {
//     Matrix matrix(rows, cols);
//     std::ifstream file(file_path);
//     std::string line;
//     int row = 0;
//     while (getline(file, line)) {
//         std::stringstream stream(line);
//         std::string cell;
//         int col = 0;
//         while (getline(stream, cell, ',')) {
//             matrix.data[row][col++] = std::stod(cell);
//         }
//         ++row;
//     }
//     file.close();
//     return matrix;
// }

// void send_matrix(double* matrix, int rows, int cols, MPI_Comm& intercomm, int rank) {
//     MPI_Send(matrix, rows * cols, MPI_DOUBLE, rank, 0, intercomm);
// }


int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);  // Initialize MPI environment

    // test new matrix multiply with summa
    // int rowsA = 4;
    // int colsA = 4;
    // int rowsB = colsA;
    // int colsB = 4;
    // Matrix A(rowsA, colsA);
    // for (int i = 0; i < rowsA; i++) {
    //     for (int j = 0; j < colsA; j++) {
    //         A.data[i][j] = i * colsA + j;
    //     }
    // }
    // Matrix B(rowsB, colsB);
    // for (int i = 0; i < rowsB; i++) {
    //     for (int j = 0; j < colsB; j++) {
    //         B.data[i][j] = (i == j) ? 1 : 0;
    //     }
    // }
    // Matrix C = Matrix::multiply(A, B);
    // std::cout << "C:\n" << C;

    // test summa
    // int rowsA = 4;
    // int colsA = 4;
    // int rowsB = colsA;
    // int colsB = 4;
    // // initialize A to be the index matrix and B to be the identity
    // double* A = new double[rowsA * colsA];
    // double* B = new double[rowsB * colsB];
    // double* C = new double[rowsA * colsB];
    // // initialize some matrices

    // for (int i = 0; i < rowsA; i++) {
    //     for (int j = 0; j < colsA; j++) {
    //         A[i * colsA + j] = i * colsA + j;
    //     }
    // }
    // for (int i = 0; i < rowsB; i++) {
    //     for (int j = 0; j < colsB; j++) {
    //         B[i * colsB + j] = (i == j) ? 1 : 0;
    //     }
    // }
    // // initialize C to be the zero matrix

    // for (int i = 0; i < rowsA; i++) {
    //     for (int j = 0; j < colsB; j++) {
    //         C[i * colsB + j] = 0;
    //     }
    // }
    // // print A, B, and C
    // std::cout << "A:\n";
    // for (int i = 0; i < rowsA; i++) {
    //     for (int j = 0; j < colsA; j++) {
    //         std::cout << A[i * colsA + j] << " ";
    //     }
    //     std::cout << std::endl;
    // }
    // std::cout << "B:\n";
    // for (int i = 0; i < rowsB; i++) {
    //     for (int j = 0; j < colsB; j++) {
    //         std::cout << B[i * colsB + j] << " ";
    //     }
    //     std::cout << std::endl;
    // }
    // std::cout << "C:\n";
    // for (int i = 0; i < rowsA; i++) {
    //     for (int j = 0; j < colsB; j++) {
    //         std::cout << C[i * colsB + j] << " ";
    //     }
    //     std::cout << std::endl;
    // }

    // int panel_width = 2; // Panel width for block decomposition
    // int num_comm_row = 2; // Number of rows in processor grid
    // int num_comm_col = 2; // Number of columns in processor grid
    // int np = num_comm_row * num_comm_col; // Total number of processes (2x2 grid)


    // // MPI_Comm intercomm;
    // // int errcodes[np];  // Array to hold error codes for each spawned process
    // // char errstr[MPI_MAX_ERROR_STRING];  // Buffer to hold the error string
    // // int resultlen;  // Variable to hold the length of the error string

    // // std::cout << "Spawning processes with parameters:\n";
    // // for (int i = 0; i < 7; i++) {
    // //     std::cout << "arg[" << i << "]: " << args[i] << std::endl;
    // // }

    // // int spawn_result = MPI_Comm_spawn("summa", args, np, MPI_INFO_NULL, 0, MPI_COMM_WORLD, &intercomm, errcodes);
    // // if (spawn_result != MPI_SUCCESS) {
    // //     printf("Failed to spawn MPI processes\n");
    // //     for (int i = 0; i < np; i++) {
    // //         if (errcodes[i] != MPI_SUCCESS) {
    // //             MPI_Error_string(errcodes[i], errstr, &resultlen);
    // //             printf("Error spawning process %d: %s\n", i, errstr);
    // //         }
    // //     }
    // //     MPI_Abort(MPI_COMM_WORLD, spawn_result);
    // // }
    // // printf("MPI processes spawned successfully\n");

    // MPI_Comm intercomm;
    // char arg1[10], arg2[10], arg3[10], arg4[10], arg5[10], arg6[10];

    // sprintf(arg1, "%d", rowsA);
    // sprintf(arg2, "%d", colsA);
    // sprintf(arg3, "%d", colsB);
    // sprintf(arg4, "%d", panel_width);
    // sprintf(arg5, "%d", num_comm_row);
    // sprintf(arg6, "%d", num_comm_col);

    // char *args[] = {"summa", arg1, arg2, arg3, arg4, arg5, arg6, NULL};

    // int errcodes[4];
    // printf("rowsA: %d, colsA: %d, colsB: %d\n", rowsA, colsA, colsB);
    // MPI_Comm_spawn("./summa", args, 4, MPI_INFO_NULL, 0, MPI_COMM_WORLD, &intercomm, errcodes);
    // printf("MPI processes spawned successfully\n");

    // // Send matrices A, B, C to the master process (rank 0 of the spawned processes)
    // send_matrix(A, rowsA, colsA, intercomm, 0);
    // send_matrix(B, rowsB, colsB, intercomm, 0);
    // send_matrix(C, rowsA, colsB, intercomm, 0);

    // // Receive the result matrix C from the master process
    // // Assuming matrix C is already allocated and ready to receive data
    // MPI_Recv(C, rowsA * colsB, MPI_DOUBLE, MPI_ANY_SOURCE, 0, intercomm, MPI_STATUS_IGNORE);

    // // Print received matrix C for verification
    // std::cout << "Received Matrix C:\n";
    // for (int i = 0; i < rowsA; i++) {
    //     for (int j = 0; j < colsB; j++) {
    //         std::cout << C[i * colsB + j] << " ";
    //     }
    //     std::cout << std::endl;
    // }
    
    // // free memory
    // delete[] A;
    // delete[] B;
    // delete[] C;

    

    int num_heads = 4;
    int d_model = 1024;
    int seq_length = 128;

    std::cout << "Running MultiHeadAttention with num_heads = " << num_heads
              << ", d_model = " << d_model << ", seq_length = " << seq_length << std::endl;

    MultiHeadAttention mha(num_heads, d_model);

    // Load weights, biases, input and expected output from CSV files
    Matrix Wq = load_csv("../input/Wq.csv", d_model, d_model);
    Matrix Wk = load_csv("../input/Wk.csv", d_model, d_model);
    Matrix Wv = load_csv("../input/Wv.csv", d_model, d_model);
    Matrix Wo = load_csv("../input/Wo.csv", d_model, d_model);

    Matrix bq = load_csv("../input/bq.csv", 1, d_model);
    Matrix bk = load_csv("../input/bk.csv", 1, d_model);
    Matrix bv = load_csv("../input/bv.csv", 1, d_model);
    Matrix bo = load_csv("../input/bo.csv", 1, d_model);

    Matrix x = load_csv("../input/mha_input.csv", seq_length, d_model);
    Matrix expected_output = load_csv("../input/mha_output.csv", seq_length, d_model);


    // print weights
    // std::cout << "Wq:\n" << Wq;
    // std::cout << "Wk:\n" << Wk;
    // std::cout << "Wv:\n" << Wv;
    // std::cout << "Wo:\n" << Wo;


    // Set weights
    mha.set_weights(Wq, Wk, Wv, Wo);

    // print bias
    // std::cout << "bq:\n" << bq << std::endl;
    // std::cout << "bk:\n" << bk << std::endl;
    // std::cout << "bv:\n" << bv << std::endl;
    // std::cout << "bo:\n" << bo << std::endl;

    // Set bias
    mha.set_biases(bq, bk, bv, bo); 

    // Print input matrix
    // std::cout << "Input:\n" << x << std::endl;

    // run mha 10 times and take average time
    int num_runs = 10;
    double total_duration = 0;
    for (int i = 0; i < num_runs; ++i) {
        std::cout << "Run " << i + 1 << " of " << num_runs << std::endl;
        auto start = std::chrono::high_resolution_clock::now();
        Matrix output = mha.compute(x);
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> duration = end - start;
        total_duration += duration.count();
        std::cout << "Execution time: " << duration.count() << " seconds\n";
    }
    std::cout << "Average execution time: " << total_duration / num_runs << " seconds\n";

    // auto start = std::chrono::high_resolution_clock::now();

    // // Compute output
    Matrix output = mha.compute(x);

    // auto end = std::chrono::high_resolution_clock::now();
    // std::chrono::duration<double> duration = end - start;

    // // Print results
    // // std::cout << "Output:\n" << output << std::endl;
    // std::cout << "Execution time: " << duration.count() << " seconds\n";

    // Print expected output
    // std::cout << "Expected output:\n" << expected_output << std::endl;

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

    MPI_Finalize();
    return 0;
}
