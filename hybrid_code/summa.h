#ifndef SUMMA_H
#define SUMMA_H

#include "mpi.h"

#ifdef __cplusplus
extern "C" {
#endif

#define min(x, y) ((x) < (y) ? (x) : (y))

void create_cart_grid(int row, int col, int *coords, MPI_Comm* CART_COMM);
void cart_sub(MPI_Comm CART_COMM, MPI_Comm* ROW_COMM, MPI_Comm* COL_COMM);
void initializeAB(int rowsA, int colsA, int rowsB, int colsB, double **A, double **B);
double* zeroInit(int rows, int cols);
double* indexInit(int rows, int cols);
double* identityInit(int size);
void divideInteger(int m, int num_comm_row, int result[]);
void getRowCol(int index, int rows, int cols, int *row, int *col);
void getIndex(int row, int col, int columns, int *index);
void blockRowCol_to_matrixRowCol(int block_row, int block_col, int *blocks_m, int *blocks_n, int *matrix_row, int *matrix_col);
double** decompose_matrix(double *matrix, int rows, int cols, int num_blocks_y, int num_blocks_x, int* blocks_m, int* blocks_n);
void matrix_multiply(int m, int n, int k, double* A, double* B, double* C);
void print_matrix(double *matrix, int rows, int cols);

int summa(int argc, char *argv[], int rowsA, int colsA, int colsB, int panel_width, int num_comm_row, int num_comm_col, double *A, double *B, double *C);


#ifdef __cplusplus
}
#endif

#endif // SUMMA_H
