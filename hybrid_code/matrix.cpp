#include "matrix.h"
#include <omp.h>
#include "mpi.h"

void send_matrix(double* matrix, int rows, int cols, MPI_Comm& intercomm, int rank) {
    MPI_Send(matrix, rows * cols, MPI_DOUBLE, rank, 0, intercomm);
}

// add a default constructor
Matrix::Matrix() : rows(0), cols(0) {}

Matrix::Matrix(int rows, int cols) : rows(rows), cols(cols), data(rows, std::vector<double>(cols, 0.0)) {}

Matrix Matrix::multiply(const Matrix& a, const Matrix& b) {
    if (a.cols != b.rows) throw std::invalid_argument("Incompatible dimensions for multiplication");
    double* A_flat = flatten_data(a.data, a.rows, a.cols);
    double* B_flat = flatten_data(b.data, b.rows, b.cols);
    double* C_flat  = new double[a.rows * b.cols];

    int rowsA = a.rows;
    int colsA = a.cols;
    int rowsB = colsA;
    int colsB = b.cols;

    int panel_width = (int) rowsA / 10; // Panel width for block decomposition
    int num_comm_row = 2; // Number of rows in processor grid
    int num_comm_col = 2; // Number of columns in processor grid
    int np = num_comm_row * num_comm_col; // Total number of processes (2x2 grid)

    MPI_Comm intercomm;
    char arg1[10], arg2[10], arg3[10], arg4[10], arg5[10], arg6[10];

    sprintf(arg1, "%d", rowsA);
    sprintf(arg2, "%d", colsA);
    sprintf(arg3, "%d", colsB);
    sprintf(arg4, "%d", panel_width);
    sprintf(arg5, "%d", num_comm_row);
    sprintf(arg6, "%d", num_comm_col);

    char *args[] = {"summa", arg1, arg2, arg3, arg4, arg5, arg6, NULL};

    int errcodes[4];
    // printf("rowsA: %d, colsA: %d, colsB: %d\n", rowsA, colsA, colsB);
    MPI_Comm_spawn("./summa", args, 4, MPI_INFO_NULL, 0, MPI_COMM_WORLD, &intercomm, errcodes);
    // printf("MPI processes spawned successfully\n");

    // Send matrices A, B, C to the master process (rank 0 of the spawned processes)
    send_matrix(A_flat, rowsA, colsA, intercomm, 0);
    send_matrix(B_flat, rowsB, colsB, intercomm, 0);
    send_matrix(C_flat, rowsA, colsB, intercomm, 0);

    // Receive the result matrix C from the master process
    // Assuming matrix C is already allocated and ready to receive data
    MPI_Recv(C_flat, rowsA * colsB, MPI_DOUBLE, MPI_ANY_SOURCE, 0, intercomm, MPI_STATUS_IGNORE);

    // Print received matrix C for verification
    // std::cout << "Received Matrix C:\n";
    // for (int i = 0; i < rowsA; i++) {
    //     for (int j = 0; j < colsB; j++) {
    //         std::cout << C_flat[i * colsB + j] << " ";
    //     }
    //     std::cout << std::endl;
    // }

    // Construct the result matrix from the flattened data
    Matrix result = construct_from_flattened_data(C_flat, rowsA, colsB);
    // printf("Result matrix constructed\n");
    // std::cout << result << std::endl;

    // free memory
    delete[] A_flat;
    delete[] B_flat;
    delete[] C_flat;

    return result;
}

double* Matrix::flatten_data(const std::vector<std::vector<double> >& matrix_data, int rows, int cols) {
    double* flat_data = new double[rows * cols];
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            flat_data[i * cols + j] = matrix_data[i][j];
        }
    }
    return flat_data;
}

Matrix Matrix::construct_from_flattened_data(double* flat_data, int rows, int cols) {
    Matrix result(rows, cols);
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            result.data[i][j] = flat_data[i * cols + j];
        }
    }
    return result;
}


Matrix Matrix::random(int rows, int cols) {
    Matrix result(rows, cols);
    for (auto& row : result.data) {
        for (auto& val : row) {
            val = static_cast<double>(rand()) / RAND_MAX;  // Fill with random values
        }
    }
    return result;
}

Matrix Matrix::transpose() const {
    Matrix result(cols, rows);
    // OpenMP
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            result.data[j][i] = data[i][j];
        }
    }
    return result;
}

Matrix Matrix::block(int startRow, int startCol, int blockRows, int blockCols) const {
    if (startRow + blockRows > rows || startCol + blockCols > cols) {
        throw std::out_of_range("Block dimensions exceed matrix dimensions.");
    }
    Matrix result(blockRows, blockCols);
    for (int i = 0; i < blockRows; ++i) {
        for (int j = 0; j < blockCols; ++j) {
            result.data[i][j] = data[startRow + i][startCol + j];
        }
    }
    return result;
}

Matrix Matrix::broadcast(int new_rows) const {
    if (rows != 1) {
        throw std::invalid_argument("Broadcast only supports matrices with a single row.");
    }
    Matrix result(new_rows, cols);
    for (int i = 0; i < new_rows; ++i) {
        result.data[i] = data[0];
    }
    return result;
}

Matrix& Matrix::operator+=(const Matrix& rhs) {
    if (this->rows != rhs.rows || this->cols != rhs.cols) {
        std::cerr << "Matrix addition error: Dimension mismatch (" 
                  << this->rows << "x" << this->cols << " vs " 
                  << rhs.rows << "x" << rhs.cols << ")." << std::endl;
        throw std::invalid_argument("Matrix dimensions must match for addition.");
    }
    for (int i = 0; i < this->rows; ++i) {
        for (int j = 0; j < this->cols; ++j) {
            this->data[i][j] += rhs.data[i][j];
        }
    }
    return *this;
}


std::ostream& operator<<(std::ostream& os, const Matrix& matrix) {
    for (int i = 0; i < matrix.rows; ++i) {
        for (int j = 0; j < matrix.cols; ++j) {
            os << matrix.data[i][j] << ' ';
        }
        os << '\n';
    }
    return os;
}
