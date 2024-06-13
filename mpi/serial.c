// #include "summa_paper.h"
#include <stdio.h>
#include <stdlib.h>
#include "mpi.h"
#include <string.h>

#define min(x, y) ((x) < (y) ? (x) : (y))
/* to compile:
brew install lapack
brew install opepanel_widthlas

mpicc -o main main.c summa_paper.c -I/opt/homebrew/opt/lapack/include -I/opt/homebrew/opt/opepanel_widthlas/include -L/opt/homebrew/opt/lapack/lib -L/opt/homebrew/opt/opepanel_widthlas/lib -llapack -lopepanel_widthlas -lm

*/

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
    MPI_Cart_sub(CART_COMM, (int[]){1, 0}, ROW_COMM);
    MPI_Cart_sub(CART_COMM, (int[]){0, 1}, COL_COMM);
}

void initializeAB(int rowsA, int colsA, int rowsB, int colsB, double **A, double **B) {
    // Allocate memory for matrices A and B
    *A = (double *)malloc(rowsA * colsA * sizeof(double));
    *B = (double *)malloc(rowsB * colsB * sizeof(double));

    if (*A == NULL || *B == NULL) {
        printf( "Memory allocation failed\n");
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

double* zeroInit(int rows, int cols) {
    double *C = (double *)malloc(rows * cols * sizeof(double));
    if (C == NULL) {
        printf("Memory allocation failed\n");
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

double* indexInit(int rows, int cols){
    double *C = (double*)malloc(rows * cols * sizeof(double));
    for (int row = 0; row < rows; row++) {
        for (int col = 0; col < cols; col++) {
            int index = row * cols + col;
            C[index] = index;
        }
    }
    return C;
}

double* identityInit(int size) {
    double *I = (double *)malloc(size * size * sizeof(double));
    if (I == NULL) {
        printf("Memory allocation failed\n");
        exit(1);
    }

    // Initialize matrix I to identity matrix in column-major order
    for (int col = 0; col < size; col++) {
        for (int row = 0; row < size; row++) {
            if (row == col)
                I[col * size + row] = 1.0; // Set diagonal elements to 1
            else
                I[col * size + row] = 0.0; // Set off-diagonal elements to 0
        }
    }

    return I;
}

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

// 1D index to row and col number
void getRowCol(int index, int rows, int cols, int *row, int *col) {
    *row = index / cols;  // Calculate row index
    *col = index % cols;  // Calculate column index
}

// row and column to 1D index
void getIndex(int row, int col, int columns, int *index){
    *index = row*columns + col;
}

// block index to matrix index (the upper left corner of the block)
void blockRowCol_to_matrixRowCol(int block_row, int block_col, int *blocks_m, int *blocks_n, int *matrix_row, int *matrix_col){
    *matrix_row = 0;
    *matrix_col = 0;
    for (int block_i=0; block_i<block_row; block_i++)
    {
        *matrix_row += blocks_m[block_i];
    }
    for (int block_j=0; block_j<block_col; block_j++)
    {
        *matrix_col += blocks_n[block_j];
    }
}

double** decompose_matrix(double *matrix, int rows, int cols, int num_blocks_y, int num_blocks_x, int* blocks_m, int* blocks_n) {
    // Calculate total number of blocks
    int num_blocks = (num_blocks_x) * (num_blocks_y);

    // Allocate memory for the result array
    double **result = (double **) malloc(num_blocks * sizeof(double));
    for (int i = 0; i < num_blocks_y; i++) {
        for (int j = 0; j < num_blocks_x; j++){
            int block_index = i * (num_blocks_x) + j;
            // printf("blocks_m[i]=%d, blocks_n[j]=%d\n", blocks_m[i], blocks_n[j]);
            (result)[block_index] = (double *) malloc(blocks_m[i] * blocks_n[j] * sizeof(double));
        }
    }

    // Decompose the matrix into blocks and flatten each block
    // block x is the x index of block
    for (int i = 0; i < num_blocks_y; i++) {
        for (int j = 0; j < num_blocks_x; j++) {
            int block_index = i * num_blocks_x + j;
            // printf("i=%d, j=%d, index=%d\n", i, j, block_index);
            double *block = (result)[block_index];
            int current_block_rows = blocks_m[i];
            int current_block_cols = blocks_n[j];
            for (int y = 0; y < current_block_rows; y++) {
                for (int x = 0; x < current_block_cols; x++) {
                    // find the offset
                    int offset_row, offset_col;
                    blockRowCol_to_matrixRowCol(i, j, blocks_m, blocks_n, &offset_row, &offset_col);
                    // printf("offset_row=%d, offset_col=%d\n", offset_row, offset_col);
                    int source_index;
                    int dest_index;
                    source_index = (offset_row + y) * cols + (offset_col + x);
                    dest_index = y * current_block_cols + x;
                    block[dest_index] = matrix[source_index];
                }
            }
        }
    }
    return result;
}

void matrix_multiply(int m, int n, int k, double* A, double* B, double* C) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            double sum = 0.0;
            for (int kk = 0; kk < k; kk++) {
                sum += A[i * k + kk] * B[kk * n + j];
            }
            C[i * n + j] += sum;
        }
    }
}

void print_matrix(double *matrix, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            printf("%3f ", matrix[i * cols + j]);
        }
        printf("\n");
    }
}

int MASTER = 0;

int main(int argc, char * argv[7])
{
    if (argc < 4){
        printf("Sorry bro! We already passed that test phase.\n");
        printf("The arguments should be m, k, n\n");
        exit(0);
    }

    /* get matrix size and number of processors */
    // A: (m, k)
    // B: (k, n)
    // C: (m, n)
    // Showing C = A*B
    int m = atoi(argv[1]);
    int k = atoi(argv[2]);
    int n = atoi(argv[3]);


    // terminal test case
    double *a, *b;
    double *c = zeroInit(m, n);

    /* MPI Init Starts */
    // MPI init comm cart
    MPI_Init(&argc, &argv);
    // MPI_Comm ROW_COMM, COL_COMM, CART_COMM;
    // int coords[2]; /* params to get processor grid */
    // create_cart_grid(num_comm_row, num_comm_col, coords, &CART_COMM);
    // cart_sub(CART_COMM, &ROW_COMM, &COL_COMM);

    // get total number of processors
    // int np;
    // MPI_Comm_size(CART_COMM, &np);

    // get my global index
    int me;
    MPI_Comm_rank(MPI_COMM_WORLD, &me );

    a = indexInit(m, k);
    b = indexInit(k, n);


    // time measurement
    double start, end;
    start = MPI_Wtime();

    matrix_multiply(m, n, k, a, b, c);

    end = MPI_Wtime();

    if (me==MASTER){
        printf("Time elapsed: %f\n", end-start);
    }

    // printf("MPI_Finalize\n");
    // MPI_Comm_free(&CART_COMM_WORKING);
    // MPI_Comm_free(&ROW_COMM_WORKING);
    // MPI_Comm_free(&COL_COMM_WORKING);

    // printf("Finalize at proc %d\n", me);
    MPI_Finalize();

    return 0;
}
