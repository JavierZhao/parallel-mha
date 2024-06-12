// #include "summa_paper.h"
#include <stdio.h>
#include <stdlib.h>
#include "mpi.h"

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
    if (argc < 7){
        printf("Sorry bro! We already passed that test phase.\n");
        printf("The arguments should be m, k, n, panel_width, num_comm_row, num_comm_col, and np\n");
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
    int panel_width = atoi(argv[4]);
    int num_comm_row = atoi(argv[5]);
    int num_comm_col = atoi(argv[6]);


    // terminal test case
    double *a, *b;
    double *c = zeroInit(m, n);

    /* MPI Init Starts */
    // MPI init comm cart
    MPI_Init(&argc, &argv);
    MPI_Comm ROW_COMM, COL_COMM, CART_COMM;
    int coords[2]; /* params to get processor grid */
    create_cart_grid(num_comm_row, num_comm_col, coords, &CART_COMM);
    cart_sub(CART_COMM, &ROW_COMM, &COL_COMM);

    // get total number of processors
    // int np;
    // MPI_Comm_size(CART_COMM, &np);

    // get my global index
    int me;
    MPI_Comm_rank(CART_COMM, &me );
    if (me == MASTER){
        a = indexInit(m, k);
        b = indexInit(k, n);
    }

    // time measurement
    double start, end;
    start = MPI_Wtime();

    // get my processor row and col index
    int myrow, mycol;
    MPI_Comm_rank( ROW_COMM, &myrow );
    MPI_Comm_rank( COL_COMM, &mycol );

    // MPI_Init(&argc, &argv);
    // MPI_Comm ROW_COMM_WORKING, COL_COMM_WORKING, CART_COMM_WORKING;
    // int num_comm_row=2, num_comm_col=2, coords[2]; /* params to get processor grid */
    // create_cart_grid(num_comm_row, num_comm_col, coords, &CART_COMM_WORKING);
    // cart_sub(CART_COMM_WORKING, &ROW_COMM_WORKING, &COL_COMM_WORKING);
    //
    // // get my global index
    // int me;
    // MPI_Comm_rank(CART_COMM_WORKING, &me );
    //
    // // get my processor row and col index
    // int myrow, mycol;
    // MPI_Comm_rank( ROW_COMM_WORKING, &myrow );
    // MPI_Comm_rank( COL_COMM_WORKING, &mycol );

    // int m, n, k,
    //     panel_width;            /* panel width */
    // double *a, *b;     /* Matrix A and B */
    // double       *c;         /* Matrix C */


    // init big matrix A, B, C if I am the master worker
    /* Test Cases */
    // 1. zero test
    // m = 4;
    // k = 8;
    // n = 4;
    // panel_width = 3;
    // double *a, *b;     /* Matrix A and B */
    // initializeAB(m, k, k, n, &a, &b);
    // b = zeroInit(k, n);

    // 2. identity test
    // m = 4;
    // k = 8;
    // n = 8;
    // panel_width = 3;
    // double *a, *b;     /* Matrix A and B */
    // a = indexInit(m, k);
    // b = identityInit(n);

    // 3. square test
    // m = 2;
    // k = 2;
    // n = 2;
    // panel_width = 1;
    // double a[] = {1.0, 2.0, 3.0, 4.0};
    // double b[] = {2.0, 0.0, 1.0, 2.0};

    // 4. bigger square test
    // m = 4;
    // k = 4;
    // n = 4;
    // panel_width = 2;
    // double *a = indexInit(m, k);
    // double *b = indexInit(k, n);

    // 5. large matrix
    // m = 1024;
    // k = 1024;
    // n = 1024;
    // panel_width = 256;
    // double *a = indexInit(m, k);
    // double *b = indexInit(k, n);

    // 6. rectangular example
    // m = 1;
    // k = 3;
    // n = 1;
    // panel_width = 1;
    // double a[] = {1, 2, 3};
    // double b[] = {1, 2, 3};

    // 7. non-communicative example 1
    // m = 1;
    // k = 2;
    // n = 1;
    // panel_width = 1;
    // double a[] = {1, 3};
    // double b[] = {1, 3};

    // 8. non-communicative example 2
    // m = 2;
    // k = 1;
    // n = 2;
    // panel_width = 1;
    // double a[] = {1, 3};
    // double b[] = {1, 3};

    // 9. large negative number
    // m = 2;
    // k = 2;
    // n = 2;
    // panel_width = 1;
    // double a[] = {-1, 1000, 5000, -6000};
    // double b[] = {1000, 0, -1, 2};

    // every worker initialize a c because I'm too lazy to send
    // local parts and reshape matrix of blocks
    double **a_blocks, **b_blocks;
    /* derive auxiliary varaibles */
    // number of blocks in each dimension
    int num_m_blocks = min(num_comm_row, m);
    int num_n_blocks = min(num_comm_col, n);
    int num_k_panels = (k+panel_width-1) / panel_width;

    // calculate block matrices dimensions
    int num_a_blocks = num_m_blocks * num_k_panels;
    int num_b_blocks = num_k_panels * num_n_blocks;
    int num_c_blocks = num_m_blocks * num_n_blocks;
    // printf("number of A blocks=%d\n", num_a_blocks);
    // printf("number of B blocks=%d\n", num_b_blocks);


    // initialize blcok sizes of A, B, C
    // the first element of each array is the major size, i.e, round_up(m/block_m)
    // the last element of each array is the remainder
    int m_a[num_m_blocks], n_a[num_k_panels],
        m_b[num_k_panels], n_b[num_n_blocks],
        m_c[num_m_blocks], n_c[num_n_blocks];

    // calculate the size of each A blocks
    divideInteger(m, num_m_blocks, m_a);
    divideInteger(k, num_k_panels, n_a);

    // calculate the size of each B blocks
    divideInteger(k, num_k_panels, m_b);
    divideInteger(n, num_n_blocks, n_b);

    // calculate the size of each C blocks
    divideInteger(m, num_m_blocks, m_c);
    divideInteger(n, num_n_blocks, n_c);

    // print the block sizes
    // printf("m_a: ");
    // for (int i = 0; i < num_comm_row; i++) {
    //     printf("%d ", m_a[i]);
    // }
    // printf("\n");
    //
    // printf("n_a: ");
    // for (int i = 0; i < panel_width; i++) {
    //     printf("%d ", n_a[i]);
    // }
    // printf("\n");
    //
    // printf("m_b: ");
    // for (int i = 0; i < panel_width; i++) {
    //     printf("%d ", m_b[i]);
    // }
    // printf("\n");
    //
    // printf("n_b: ");
    // for (int i = 0; i < num_comm_col; i++) {
    //     printf("%d ", n_b[i]);
    // }
    // printf("\n");
    //
    // printf("m_c: ");
    // for (int i = 0; i < num_comm_row; i++) {
    //     printf("%d ", m_c[i]);
    // }
    // printf("\n");
    //
    // printf("n_c: ");
    // for (int i = 0; i < num_comm_col; i++) {
    //     printf("%d ", n_c[i]);
    // }
    // printf("\n");


    // check if the processors at this row, col will be used
    int row_working = (myrow < num_m_blocks);
    int col_working = (mycol < num_n_blocks);
    int working = row_working && col_working;
    // printf("Is proc %d working? %d\n", me, working);

    // split MPI_COMM to only let working workers communicate
    MPI_Comm CART_COMM_WORKING, ROW_COMM_WORKING, COL_COMM_WORKING;
    MPI_Comm_split(CART_COMM, working, me, &CART_COMM_WORKING);
    MPI_Comm_split(ROW_COMM, working, myrow, &ROW_COMM_WORKING);
    MPI_Comm_split(COL_COMM, working, mycol, &COL_COMM_WORKING);

    MPI_Comm_free(&CART_COMM);
    MPI_Comm_free(&ROW_COMM);
    MPI_Comm_free(&COL_COMM);
    // CART_COMM_WORKING = CART_COMM;
    // ROW_COMM_WORKING = ROW_COMM;
    // COL_COMM_WORKING = COL_COMM;

    // MPI_Comm_rank(CART_COMM_WORKING, &me );
    // MPI_Comm_rank(ROW_COMM_WORKING, &myrow );
    // MPI_Comm_rank(COL_COMM_WORKING, &mycol );
    // printf("me=%d, myrow=%d, mycol=%d\n", me, myrow, mycol);

    // early exit if this worker is not working
    if (!working){
        MPI_Finalize();
        return 0;
    }

    // allocate memory for broadcasting row chunks of A
    int my_A_panel_num = num_k_panels/num_n_blocks;
    if (mycol < (num_k_panels%num_n_blocks)) {my_A_panel_num++;}
    // printf("My numbe of A panels=%d\n", my_A_panel_num);
    double **my_A_blocks = (double **) malloc(my_A_panel_num * sizeof(double));
    for (int block_k=0; block_k<my_A_panel_num; block_k++){
        // retrieve global panel index
        int global_panel_index = block_k*num_n_blocks + mycol;
        // debug
        // printf("Creating A block (%d, %d) of shape (%d, %d) at proc (%d, %d) block %d\n",\
            myrow, global_panel_index, m_a[myrow], n_a[global_panel_index], myrow, mycol, block_k);
        my_A_blocks[block_k] = (double *)malloc(m_a[myrow] * n_a[global_panel_index] * sizeof(double));
    }

    // allocate memory for broadcasting column chunks of B
    int my_B_panel_num = num_k_panels/num_m_blocks;
    if (myrow < (num_k_panels%num_m_blocks)) {my_B_panel_num++;}
    double **my_B_blocks = (double **) malloc(my_B_panel_num * sizeof(double));
    for (int block_k=0; block_k<my_B_panel_num; block_k++){
        // retrieve global panel index
        int global_panel_index = block_k*num_m_blocks + myrow;
        // debug
        // printf("Creating A block (%d, %d) of size %d at proc (%d, %d) block %d\n",\
        //    myrow, global_panel_index, m_a[myrow]*n_a[global_panel_index], myrow, mycol, block_k);
        my_B_blocks[block_k] = (double *)malloc(m_b[global_panel_index] * n_b[mycol] * sizeof(double));
    }

    // early exit if the worker won't be working

    MPI_Barrier(CART_COMM_WORKING);
    // master code starts
    if ( me == MASTER){

        // initializeAB(m, k, k, n, &a, &b);
        // b = identityInit(n);
        // c = zeroInit(m, n);
        // printf("Matrix A:\n");
        // print_matrix(a, m, k);

        // printf("Matrix B:\n");
        // print_matrix(b, k, n);

        // printf("Matrix C:\n");
        // print_matrix(c, m, n);

        // create block matrices
        a_blocks = decompose_matrix(a, m, k, num_m_blocks, num_k_panels, m_a, n_a);
        b_blocks = decompose_matrix(b, k, n, num_k_panels, num_n_blocks, m_b, n_b);

        // distribute blocks to workers
        // a blocks
        for (int block_idx=0; block_idx < num_a_blocks; block_idx++){
            // get the block's row and column
            int block_row, block_col;
            getRowCol(block_idx, num_m_blocks, num_k_panels, &block_row, &block_col);

            // debug
            // printf("printing A bock i=%d, row=%d, col=%d\n", block_idx, m_a[block_row], n_a[block_col]);
            // print_matrix(a_blocks[block_idx], m_a[block_row], n_a[block_col]);

            // derive the destiny processor's row and column
            int comm_row, comm_col;
            comm_row = block_row;
            comm_col = block_col % num_n_blocks;

            // derive the destiny processor's 1D index
            int dest_comm_index;
            getIndex(comm_row, comm_col, num_n_blocks, &dest_comm_index);

            // int MPI_Isend(const void *buf, int count, MPI_Datatype datatype, int dest,
            //  int tag, MPI_Comm comm, MPI_Request *request)
            // printf("Master is sending sending A block (%d, %d) of shape (%d, %d) to proc (%d, %d)\n",
            //       block_row, block_col, m_a[block_row], n_a[block_col], comm_row, comm_col);
            // printf("sending buffer pointer is null? %lf\n", *a_blocks[block_idx]);
            // printf("Send A block %d to dest %d\n", block_idx, dest_comm_index);
            MPI_Request request;
            MPI_Isend(a_blocks[block_idx], m_a[block_row]*n_a[block_col], MPI_DOUBLE,
                     dest_comm_index, block_idx, CART_COMM_WORKING, &request);
        }

        // b blocks
        for (int block_idx=0; block_idx < num_b_blocks; block_idx++){
            // get the block's row and column
            int block_row, block_col;
            getRowCol(block_idx, num_k_panels, num_n_blocks, &block_row, &block_col);

            // derive the destiny processor's row and column
            int comm_row, comm_col;
            comm_row = block_row % num_m_blocks;
            comm_col = block_col;

            // derive the destiny processor's 1D index
            int dest_comm_index;
            getIndex(comm_row, comm_col, num_n_blocks, &dest_comm_index);

            // int MPI_Isend(const void *buf, int count, MPI_Datatype datatype, int dest,
            //  int tag, MPI_Comm comm, MPI_Request *request)
            // printf("Sending size %d\n", m_b[block_row]*n_b[block_col]);
            // printf("sending buffer pointer is null? %lf\n", *b_blocks[block_idx]);
            MPI_Request request;
            MPI_Isend(b_blocks[block_idx], m_b[block_row]*n_b[block_col], MPI_DOUBLE,
                     dest_comm_index, block_idx+num_a_blocks , CART_COMM_WORKING, &request);

             // Debugging
            // printf("Send B block %d to dest %d\n", block_idx, dest_comm_index);
        }

    }

    // debug recieving A blocks
    // printf("my_A_panel_num=%d\n", my_A_panel_num);
    // printf("my_m_a=%d\n", m_a[myrow]);
    // printf("my_n_a=%d\n", n_a[mycol]);
    // printf("Start recieve\n");

    // worker code starts
    // receieve my A blocks

    for (int block_k=0; block_k<my_A_panel_num; block_k++){
        // printf("my A block_k=%d\n", block_k);
        // retrieve the global index
        int source_a_block_idx, source_a_block_row, source_a_block_col;
        source_a_block_row = myrow;
        source_a_block_col = block_k*num_n_blocks + mycol;
        getIndex(source_a_block_row, source_a_block_col, num_k_panels, &source_a_block_idx);

        // debug
        // printf("Recieve A block %d at worker %d\n", source_a_block_idx, me);
        // printf("Recieveing size %d\n", m_a[myrow]*n_a[block_k]);
        // printf("buffer pointer is null? %lf\n", *my_A_blocks[block_k]);
        // printf("proc (%d, %d) is recieving A block (%d, %d) of size %d from Master\n",
        //         myrow, mycol, source_a_block_row, source_a_block_col, m_a[source_a_block_row]*n_a[source_a_block_col]);
        MPI_Request request;
        MPI_Irecv(my_A_blocks[block_k], m_a[source_a_block_row]*n_a[source_a_block_col], MPI_DOUBLE,
                    MASTER, source_a_block_idx, CART_COMM_WORKING, &request);
        MPI_Wait(&request, MPI_STATUS_IGNORE);
    }
    MPI_Barrier(CART_COMM_WORKING);

    // debug recieving B blocks
    // printf("my_B_panel_num=%d\n", my_B_panel_num);
    // printf("my_m_b=%d\n", m_b[myrow]);
    // printf("my_n_b=%d\n", n_b[mycol]);
    // printf("Start recieve\n");

    // recieve my B blocks
    // debug
    // printf("start receive b blocks");
    for (int block_k=0; block_k<my_B_panel_num; block_k++){
        // retrieve the global index
        int source_b_block_idx, source_b_block_row, source_b_block_col;
        source_b_block_row = block_k*num_m_blocks + myrow;
        source_b_block_col = mycol;
        getIndex(source_b_block_row, source_b_block_col, num_n_blocks, &source_b_block_idx);

        // debug
        // printf("Recieve B block %d at worker %d\n", source_b_block_idx, me);
        // printf("Recieveing size %d\n", m_b[block_k]*n_b[mycol]);
        // printf("buffer pointer is null? %lf\n", *my_B_blocks[block_k]);
        MPI_Request request;
        MPI_Irecv(my_B_blocks[block_k], m_b[source_b_block_row]*n_b[source_b_block_col], MPI_DOUBLE,
                    MASTER, source_b_block_idx+num_a_blocks, CART_COMM_WORKING, &request);
        MPI_Wait(&request, MPI_STATUS_IGNORE);
    }
    MPI_Barrier(CART_COMM_WORKING);

    /* debug purpose */
    /* check if all blocks are correctly sent and received */
    // for (int block_k=0; block_k<my_A_panel_num; block_k++){
    //     printf("proc_row=%d, proc_col=%d, panel_i=%d\n", myrow, mycol, block_k);
    //     source_a_block_col = block_k*num_comm_col + mycol;
    //     print_matrix(my_A_blocks[block_k], m_a[myrow], n_a[source_a_block_col]);
    // }

    // for (int block_k=0; block_k<my_B_panel_num; block_k++){
    //     printf("proc_row=%d, proc_col=%d, panel_i=%d\n", myrow, mycol, block_k);
    //     // local panel index to global panel index
    //     source_b_block_row = block_k*num_comm_row + myrow;
    //     print_matrix(my_B_blocks[block_k], m_b[source_b_block_row], n_b[mycol]);
    // }

    /* SUMMA starts */
    // printf("Start SUMMA\n");

    double *workC = (double *)malloc(m_c[myrow] * n_c[mycol] * sizeof(double));
    // printf("me=%d, myrow=%d, mycol=%d\n", me, myrow, mycol);
    for (int panel_k=0; panel_k<num_k_panels; panel_k++){
        // I am a worker processsor
        // I am looking for the processor that is in the same row with me
        // and has the block A for brodacasting
        int row_root = panel_k % num_n_blocks; // this will return a column index

        // find where the row root worker stores this block
        // aka the index in my_A_blocks
        int row_root_block_index = panel_k/num_n_blocks;

        // copy block A to the working area if I am the row root processor
        double *workA = (double *)malloc(m_a[myrow] * n_a[panel_k] * sizeof(double));
        if (mycol == row_root){
            // printf("assigning A block (%d, %d) of size %d from proc (%d, %d) block %d\n",\
             myrow, panel_k, m_a[myrow]*n_a[panel_k], myrow, mycol, row_root_block_index);
            workA = my_A_blocks[row_root_block_index];
        }

        // Now I am looking for the processor that is in the same column with me
        // and has the block B for broadcasting
        int col_root = panel_k % num_m_blocks; // this will return a row index

        // find where the col root worker stores this block
        // aka the index in my_B_blocks
        int col_root_block_index = panel_k/num_m_blocks;
        // printf("I am panel_k=%d, myrow=%d, mycol=%d, col_root=%d, col_root_block_index=%d\n", panel_k, myrow, mycol, col_root, col_root_block_index);

        // copy block B to the working area if I am the col root processor
        double *workB = (double *)malloc(m_b[panel_k] * n_b[mycol] * sizeof(double));
        if (myrow == col_root){
            workB = my_B_blocks[col_root_block_index];
        }

        // broadcase block_A and block_B
        MPI_Bcast(workA, m_a[myrow]*n_a[panel_k], MPI_DOUBLE, row_root, COL_COMM_WORKING);
        MPI_Bcast(workB, m_b[panel_k]*n_b[mycol], MPI_DOUBLE, col_root, ROW_COMM_WORKING);

        // Calculate block_C += block_A * block_B
        matrix_multiply(m_c[myrow], n_c[mycol], n_a[panel_k], workA, workB, workC);

        // free(workA);
        // free(workB);
    }

    // write workC to the local block of the matrix C
    // every blocks other than workC will be zeros
    int row_offset, col_offset;
    blockRowCol_to_matrixRowCol(myrow, mycol, m_c, n_c, &row_offset, &col_offset);
    int C_matrix_i, C_matrix_j, C_matrix_index;
    int workC_index;
    for (int i = 0; i < m_c[myrow]; i++) {
        for (int j = 0; j < n_c[mycol]; j++){
            workC_index = i * n_c[mycol] + j;
            C_matrix_i = row_offset+i;
            C_matrix_j = col_offset+j;
            C_matrix_index = C_matrix_i * n + C_matrix_j;

            c[C_matrix_index] = workC[workC_index];
        }
    }

    // sum all local matrix C
    // int MPI_Reduce(const void *sendbuf, void *recvbuf, int count, MPI_Datatype datatype,
    //           MPI_Op op, int root, MPI_Comm comm)
    double *c_result = zeroInit(m, n);
    MPI_Barrier(CART_COMM_WORKING);
    MPI_Reduce(c, c_result, m*n, MPI_DOUBLE, MPI_SUM, MASTER, CART_COMM_WORKING);

    end = MPI_Wtime();
    if (me == MASTER){
        printf("Time elapsed: %f\n", end-start);
    }

    // MASTER worker print the finale result
    // if (me == MASTER){
    //     printf("Matrix result C:\n");
    //     print_matrix(c_result, m, n);
    //     printf("Finish printing\n");
    // }





    // call summa
    // summa(m, n, k, panel_width, a, lda, b, ldb, c, ldc, m_a, n_a, m_b, n_b, m_c, n_c, ROW_COMM, COL_COMM);
    // MPI_Barrier(MPI_COMM_WORLD);
    // // For Debugging: directly compute C = A*B
    // // for (int i = 0; i < m; i++) {
    // //     for (int j = 0; j < n; j++) {
    // //         for (int l = 0; l < k; l++) {
    // //             c[j * m + i] += a[l * m + i] * b[j * k + l];
    // //         }
    // //     }
    // // }




    // MPI_Barrier(CART_COMM_WORKING);
    // MPI_Barrier(ROW_COMM_WORKING);
    // MPI_Barrier(COL_COMM_WORKING);

    free(a);
    free(b);
    free(c);
    free(c_result);

    for (int block_k=0; block_k<my_A_panel_num; block_k++){
        free(my_A_blocks[block_k]);
    }
    free(my_A_blocks);

    for (int block_k=0; block_k<my_B_panel_num; block_k++){
        free(my_B_blocks[block_k]);
    }
    free(my_B_blocks);

    if (me==MASTER){
        for (int i=0; i<num_a_blocks; i++){
            free(a_blocks[i]);
        }
        free(a_blocks);

        for (int i=0; i<num_b_blocks; i++){
            free(b_blocks[i]);
        }
        free(b_blocks);
    }

    MPI_Barrier(CART_COMM_WORKING);
    MPI_Barrier(ROW_COMM_WORKING);
    MPI_Barrier(COL_COMM_WORKING);

    // printf("MPI_Finalize\n");
    // MPI_Comm_free(&CART_COMM_WORKING);
    // MPI_Comm_free(&ROW_COMM_WORKING);
    // MPI_Comm_free(&COL_COMM_WORKING);

    // printf("Finalize at proc %d\n", me);
    MPI_Finalize();

    return 0;
}
