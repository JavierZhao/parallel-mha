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

void cyclic_distribution(MPI_Comm CART_COMM, int menum, int source,
                         int num_proc_row, int num_proc_col,
                         int n, int m, int p, /* n: number of rows? p: number of columns */
                         int lda, int ldb, int ldc,
                         float *A, float *B, float *C,
                         float **locA, float **locB, float **locC)
{
    float *bufferA, *bufferB, *tmpA, *tmpB, *tmpC;

    int dest;
    int local_offset, offsetA, offsetB;

    /* Calcualte the size of local matrices */

    // Size of individual blocks
    int n_size = n / num_proc_row;    // numero di righe
    int p_size = p / num_proc_col;    // numero di colonne

    // calcolo del numero di blocchi
    // sulle colonne di A e sulle righe di B
    int blk_num = lcm(num_proc_row, num_proc_col);
    int proc_blk_a = blk_num / num_proc_col;
    int proc_blk_b = blk_num / num_proc_row;

    // numero di colonne di un blocco di A
    // equivalente al numero di righe di un blocco di B
    int m_size = m / blk_num;

    int blk_size_a = n_size * m_size;
    int blk_size_b = m_size * p_size;
    int blk_size_c = n_size * p_size;

    // allocazione matrici locali
    tmpA = (float*) malloc(sizeof(float) *
                          (n_size * m_size) * proc_blk_a);

    tmpB = (float*) malloc(sizeof(float) *
                          (m_size * proc_blk_b) * p_size);

    tmpC = (float*) calloc(blk_size_c, sizeof(float));

    // allocazione matrici di supporto per l'invio
    bufferA = (float*) malloc(sizeof(float) * blk_size_a);
    bufferB = (float*) malloc(sizeof(float) * blk_size_b);

    /* distribuzione ciclica delle matrici */

    local_offset = 0;

    MPI_Barrier(MPI_COMM_WORLD);

    // distribuzione di A
    for (int i = 0; i < num_proc_row; i++)
    {
        for (int j = 0; j < blk_num; j++)
        {
            // calcolo id destinatario
            dest = (i * num_proc_col) + (j % num_proc_col);
            offsetA = (i * lda * n_size) + (j * m_size);

            if (menum == source)
            {
                if(dest == source)
                {
                    cp_matrix(&A[offsetA], &tmpA[local_offset*m_size], n_size, m_size, lda, m_size * proc_blk_a);
                    local_offset++;
                }
                else
                {
                    cp_matrix(&A[offsetA], bufferA, n_size, m_size, lda, m_size);
                    MPI_Send(bufferA, blk_size_a, MPI_FLOAT, dest, 1, CART_COMM);
                }
            }
            else if (menum == dest)
            {
                MPI_Recv(bufferA, blk_size_a, MPI_FLOAT, source, 1, CART_COMM, MPI_STATUS_IGNORE);
                cp_matrix(bufferA, &tmpA[local_offset*m_size], n_size, m_size, m_size, m_size * proc_blk_a);
                local_offset++;
            }
        }
    }

    local_offset = 0;

    // distrubuzione di B
    for (int i = 0; i < blk_num; i++)
    {
        for (int j = 0; j < num_proc_col; j++)
        {
            // calcolo id destinatario
            dest = ((i % num_proc_row)*num_proc_col) + j;
            offsetB = (i * ldb * m_size) + (j * p_size);

            if(menum == source)
            {
                if(dest == source)
                {
                    cp_matrix(&B[offsetB], &tmpB[local_offset*blk_size_b], m_size, p_size, ldb, p_size);
                    local_offset++;
                }
                else
                {
                    cp_matrix(&B[offsetB], bufferB, m_size, p_size, ldb, p_size);
                    MPI_Send(bufferB, blk_size_b, MPI_FLOAT, dest, 1, CART_COMM);
                }
            }
            else if(menum == dest)
            {
                MPI_Recv(bufferB, blk_size_b, MPI_FLOAT, source, 1, CART_COMM, MPI_STATUS_IGNORE);
                cp_matrix(bufferB, &tmpB[local_offset*blk_size_b], m_size, p_size, p_size, p_size);
                local_offset++;
            }
        }
    }

    *locA = tmpA;
    *locB = tmpB;
    *locC = tmpC;

    free(bufferA);
    free(bufferB);
}
