#ifndef MATRIX_H
#define MATRIX_H

#include <vector>
#include <iostream>
#include <cmath>

class Matrix {
public:
    std::vector<std::vector<double> > data;
    int rows, cols;

    Matrix(int rows, int cols);
    Matrix();  // Add this line
    static Matrix multiply(const Matrix& a, const Matrix& b);
    static Matrix random(int rows, int cols);
    Matrix transpose() const;
    Matrix block(int startRow, int startCol, int blockRows, int blockCols) const;
    Matrix broadcast(int new_rows) const;  // Add this line

    friend std::ostream& operator<<(std::ostream& os, const Matrix& matrix);
    Matrix& operator+=(const Matrix& rhs);

};

#endif // MATRIX_H
