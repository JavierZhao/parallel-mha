#include "matrix.h"
#include <omp.h>
// add a default constructor
Matrix::Matrix() : rows(0), cols(0) {}

Matrix::Matrix(int rows, int cols) : rows(rows), cols(cols), data(rows, std::vector<double>(cols, 0.0)) {}

Matrix Matrix::multiply(const Matrix& a, const Matrix& b) {
    if (a.cols != b.rows) throw std::invalid_argument("Incompatible dimensions for multiplication");
    Matrix result(a.rows, b.cols);
<<<<<<< HEAD
    #pragma omp parallel for
    for (int i = 0; i < result.rows; ++i) {
        for (int j = 0; j < result.cols; ++j) {
            for (int k = 0; k < a.cols; ++k) {
                result.data[i][j] += a.data[i][k] * b.data[k][j];
            }
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
