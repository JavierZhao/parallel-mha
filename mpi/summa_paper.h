#ifndef SUMMA_H
#define SUMMA_H

#include "mpi.h"

/* Macros for accessing matrix elements in column-major order */
#define A(i, j) (a[(j)*lda + (i)])
#define B(i, j) (b[(j)*ldb + (i)])
#define C(i, j) (c[(j)*ldc + (i)])

/* Macro for calculating the minimum of two values */
#define min(x, y) ((x) < (y) ? (x) : (y))

/* Global constants for BLAS calls */
extern int i_one;
extern double d_one, d_zero;

/* Function prototypes */
void summa(int m, int n, int k, int nb,
           double *a, int lda, double *b, int ldb, double *c, int ldc,
           int m_a[], int n_a[], int m_b[], int n_b[], int m_c[], int n_c[],
           MPI_Comm comm_row, MPI_Comm comm_col);
void matrix_multiply(int m, int n, int k, double *A, int lda, double *B, int ldb, double *C, int ldc);
void RING_Bcast(double *buf, int count, MPI_Datatype type, int root, MPI_Comm comm);

extern void dlacpy_(const char *uplo, const int *m, const int *n, const double *a, const int *lda, double *b, const int *ldb);
extern void dgemm_(const char *transa, const char *transb, const int *m, const int *n, const int *k, const double *alpha, const double *a, const int *lda, const double *b, const int *ldb, const double *beta, double *c, const int *ldc);

#endif /* SUMMA_H */
