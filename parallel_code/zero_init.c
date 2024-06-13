#include <stdio.h>
#include <stdlib.h>

// Function to initialize a matrix with zeros
double* zeroInit(int rows, int cols) {
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

int main() {
    int m = 2; // Example number of rows
    int n = 2; // Example number of columns

    // Initialize matrix C with zeros
    double *c = zeroInit(m, n);

    // Print matrix C to verify it is zero-initialized
    printf("Matrix C initialized to zero:\n");
    for (int col = 0; col < n; col++) {
        for (int row = 0; row < m; row++) {
            printf("%.1f ", c[col * m + row]);
        }
        printf("\n");
    }

    // Free allocated memory
    free(c);

    return 0;
}
