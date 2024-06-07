// #include "summa_paper.h"
#include <stdio.h>
#include <stdlib.h>
#include "mpi.h"

void divideInteger(int m, int num_comm_row, int result[]) {
    int quotient = m / num_comm_row;
    int remainder = m % num_comm_row;

    // Distribute the quotient evenly
    for (int i = 0; i < num_comm_row; i++) {
        result[i] = quotient;
    }

    // Distribute the remainder
    for (int i = 0; i < remainder; i++) {
        result[i]++;
    }
}


void create_cart_grid(int row, int col, int *coords, MPI_Comm* CART_COMM)
{
    int menum_cart;

    MPI_Cart_create(MPI_COMM_WORLD, 2, (int[]){row, col},
                    (int[]){1, 1}, 1, CART_COMM);

    MPI_Comm_rank(*CART_COMM, &menum_cart);

    MPI_Cart_coords(*CART_COMM, menum_cart, 2, coords);
}

void cart_sub(MPI_Comm CART_COMM, MPI_Comm* ROW_COMM,MPI_Comm* COL_COMM)
{
    MPI_Cart_sub(CART_COMM, (int[]){0, 1}, ROW_COMM);
    MPI_Cart_sub(CART_COMM, (int[]){1, 0}, COL_COMM);
}

void initializeAB(int rowsA, int colsA, int rowsB, int colsB, double **A, double **B) {
    // Allocate memory for matrices A and B
    *A = (double *)malloc(rowsA * colsA * sizeof(double));
    *B = (double *)malloc(rowsB * colsB * sizeof(double));

    if (*A == NULL || *B == NULL) {
        fprintf(stderr, "Memory allocation failed\n");
        exit(1);
    }

    // Initialize matrices A and B to 1 in column-major order
    for (int col = 0; col < colsA; col++) {
        for (int row = 0; row < rowsA; row++) {
            (*A)[col * rowsA + row] = 1.0;
        }
    }

    for (int col = 0; col < colsB; col++) {
        for (int row = 0; row < rowsB; row++) {
            (*B)[col * rowsB + row] = 1.0;
        }
    }
}

double* zeroInitC(int rows, int cols) {
    double *C = (double *)malloc(rows * cols * sizeof(double));
    if (C == NULL) {
        fprintf(stderr, "Memory allocation failed\n");
        exit(1);
    }

    // Initialize matrix C to 0 in column-major order
    for (int col = 0; col < cols; col++) {
        for (int row = 0; row < rows; row++) {
            C[col * rows + row] = 0.0;
        }
    }

    return C;
}
int main(int argc, char * argv[])
{
    MPI_Init(&argc, &argv);
    int m, n, k,
        nb;            /* panel width */
    double *a, *b,     /* Matrix A and B */
           *c;         /* Matrix C */
    int lda, ldb, ldc;
    int num_comm_row, num_comm_col, coords[2]; /* params to get processor grid */

    MPI_Comm ROW_COMM, COL_COMM, CART_COMM;

    create_cart_grid(num_comm_row, num_comm_col, coords, &CART_COMM);
    cart_sub(CART_COMM, &ROW_COMM, &COL_COMM);

    m = 256;
    k = 512;
    n = 128;
    nb = 32;

    initializeAB(m, k, k, n, &a, &b);
    c = zeroInitC(m, n);

    lda = m;
    ldb = k;
    ldc = m;
	
    // initialize blcok sizes of A, B, C
    int m_a[num_comm_row], n_a[nb],
    	m_b[nb], n_b[num_comm_col],
	m_c[num_comm_row], n_c[num_comm_col];


    // calculate the size of A blocks
    divideInteger(m, num_comm_row, m_a);
    divideInteger(k, nb, n_a);

    // calculate the size of B blocks
    divideInteger(k, nb, m_b);
    divideInteger(n, num_comm_col, n_b);

    // calculate the size of A blocks
    divideInteger(m, num_comm_row, m_c);
    divideInteger(n, num_comm_col, n_c);
}

// void summa( m, n, k, nb,
//             a, lda, b, ldb, c, ldc,
//             m_a, n_a, m_b, n_b, m_c, n_c,
//             comm_row, comm_col, work1, work2 )
