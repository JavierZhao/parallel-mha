#include <stdio.h>
#include <stdlib.h>

void initializeMatrices(int rowsA, int colsA, int rowsB, int colsB, double **A, double **B) {
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

int main() {
    double *A, *B;
    int rowsA = 2; // Example size
    int colsA = 4; // Example size

    int rowsB = 4; // Example size
    int colsB = 2; // Example size

    initializeMatrices(rowsA, colsA, rowsB, colsB, &A, &B);

    // Example usage: print matrix A
    printf("Matrix A:\n");
    for (int col = 0; col < colsA; col++) {
        for (int row = 0; row < rowsA; row++) {
            printf("%g ", A[col * rowsA + row]);
        }
        printf("\n");
    }

    printf("Matrix B:\n");
    for (int col = 0; col < colsB; col++) {
        for (int row = 0; row < rowsB; row++) {
            printf("%g ", B[col * rowsB + row]);
        }
        printf("\n");
    }

    // Free allocated memory
    free(A);
    free(B);

    return 0;
}
