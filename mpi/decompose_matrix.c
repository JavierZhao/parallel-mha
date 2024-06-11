#include <stdio.h>
#include <stdlib.h>

// 1D index to row and col index
void getRowCol(int index, int rows, int cols, int *row, int *col) {
    *row = index / cols;  // Calculate row index
    *col = index % cols;  // Calculate column index
}

// block index to matrix index (the upper left corner of the block)
void blockIdx_to_matrixIdx(int block_row, int block_col, int *blocks_m, int *blocks_n, int *matrix_row, int *matrix_col){
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

double** decompose_matrix(double *matrix, int rows, int cols, int block_rows, int block_cols, int* blocks_m, int* blocks_n) {
    // round up num_blocks
    int num_blocks_x = (cols+block_cols-1) / block_cols;
    int num_blocks_y = (rows+block_rows-1) / block_rows;

    // Calculate total number of blocks
    int num_blocks = (num_blocks_x) * (num_blocks_y);

    // Allocate memory for the result array
    int block_index;
    double **result = (double **) malloc(num_blocks * sizeof(double));
    for (int i = 0; i < num_blocks_y; i++) {
        for (int j = 0; j < num_blocks_x; j++){
            block_index = i * (num_blocks_x) + j;
            // printf("blocks_m[i]=%d, blocks_n[j]=%d\n", blocks_m[i], blocks_n[j]);
            (result)[block_index] = (double *) malloc(blocks_m[i] * blocks_n[j] * sizeof(double));
        }
    }

    // Decompose the matrix into blocks and flatten each block
    // block x is the x index of block
    block_index = 0;
    int source_index;
    int dest_index;
    int offset_row, offset_col;
    for (int i = 0; i < num_blocks_y; i++) {
        for (int j = 0; j < num_blocks_x; j++) {
            block_index = i * num_blocks_x + j;
            double *block = (result)[block_index];
            for (int y = 0; y < block_rows; y++) {
                for (int x = 0; x < block_cols; x++) {
                    // find the offset
                    // printf("i=%d, j=%d\n", i, j);
                    blockIdx_to_matrixIdx(i, j, blocks_m, blocks_n, &offset_row, &offset_col);
                    // printf("offset_row=%d, offset_col=%d\n", offset_row, offset_col);
                    source_index = (offset_row + y) * cols + (offset_col + x);
                    dest_index = y * blocks_n[j] + x;
                    block[dest_index] = matrix[source_index];
                }
            }
        }
    }
    return result;
}

void print_matrix(double *matrix, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            printf("%3f ", matrix[i * cols + j]);
        }
        printf("\n");
    }
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

int main() {

    // square test case
    // int rows = 4, cols = 4;
    // int num_comm_row=2, num_comm_col=2;
    // double matrix[16] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};

    // square matrix irregular decomposition test case
    // int rows = 3, cols = 3;
    // int num_comm_row=2, num_comm_col=2;
    // double matrix[9] = {1, 2, 3, 4, 5, 6, 7, 8, 9};

    // rectangular test case
    // int rows = 4, cols = 6;
    // int num_comm_row = 2, num_comm_col = 2;
    // double matrix[24] = {1, 2, 3, 4, 5, 6,
    //                     7, 8, 9, 10, 11, 12,
    //                     13, 14, 15, 16, 17, 18,
    //                     19, 20, 21, 22, 23, 24};

    // rectangular matrix irregular decomposition test case
    int rows=4, cols=5;
    int num_comm_row = 2, num_comm_col = 2;
    double matrix[20] = {1, 2, 3, 4, 5,
                            6, 7, 8, 9, 10,
                            11, 12, 13, 14, 15,
                            16, 17, 18, 19, 20};

    // row matrix test case
    // int rows=1, cols=4;
    // int num_comm_row=2, num_comm_col=2;
    // double matrix[4] = {1, 2, 3, 4};

    // column matrix test case
    // int rows=4, cols=1;
    // int num_comm_row=2, num_comm_col=2;
    // double matrix[4] = {1, 2, 3, 4};

    // derive number of rows and columns in each block
    int block_rows = (rows+num_comm_row-1) / num_comm_row;
    int block_cols = (cols+num_comm_col-1) / num_comm_col;

    // derive number of blocks in each dimension
    int num_blocks_x = (cols+block_cols-1) / block_cols;
    int num_blocks_y = (rows+block_rows-1) / block_rows;

    int blocks_m[num_blocks_y], blocks_n[num_blocks_x];
    divideInteger(rows, num_comm_row, blocks_m);
    divideInteger(cols, num_comm_col, blocks_n);

    double **result;
    printf("block_rows=%d, block_cols=%d\n", block_rows, block_cols);

    printf("blocks_m: ");
    for (int i = 0; i < num_blocks_y; i++) {
        printf("%d ", blocks_m[i]);
    }
    printf("\n");

    printf("blocks_n: ");
    for (int i = 0; i < num_blocks_x; i++) {
        printf("%d ", blocks_n[i]);
    }
    printf("\n");

    result = decompose_matrix(matrix, rows, cols, block_rows, block_cols, blocks_m, blocks_n);

    // Print each block
    int block_i, block_j;
    for (int i = 0; i < num_blocks_x * num_blocks_y; i++) {
        printf("Block %d:\n", i);
        getRowCol(i, num_blocks_y, num_blocks_x, &block_i, &block_j);
        printf("Block indices: block_i=%d, block_j=%d\n", block_i, block_j);
        printf("Print arguments: block=%d, rows=%d, cols=%d\n", i, blocks_m[block_i], blocks_n[block_j]);
        print_matrix(result[i], blocks_m[block_i], blocks_n[block_j]);
    }

    // Free allocated memory
    for (int i = 0; i < num_blocks_x * num_blocks_y; i++) {
        free(result[i]);
    }
    free(result);

    return 0;
}
