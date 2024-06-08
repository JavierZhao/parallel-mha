#include "mpi.h"
#include <stdlib.h>
#include <stdio.h>
#include "summa_paper.h"

/*We are using column major indexing*/
// #define A( i,j ) (a[ j*lda+i ])
// #define B( i,j ) (b[ j*ldb+i ])
// #define C( i,j ) (c[ j*ldc+i ])

// #define min( x,y ) ( (x) < (y) ? (x) : (y) )

int i_one = 1; /* used for constant passed to blas call */
double d_one=1.0,
        d_zero=0.0; /* used for constant passed to blas call */

void summa( m, n, k, nb,
            a, lda, b, ldb, c, ldc,
            m_a, n_a, m_b, n_b, m_c, n_c,
            comm_row, comm_col)
int m, n, k,
    nb,           /* panel width */
    m_a[], n_a[], /* dimensions of blocks of A */
    m_b[], n_b[], /* dimensions of blocks of B */
    m_c[], n_c[], /* dimensions of blocks of C */
    lda, ldb, ldc;

double *a, *b, *c;

MPI_Comm comm_row, comm_col;
{
    int myrow, mycol,
        nprow, npcol,
        i, j, kk, iwrk,
        icurrow, icurcol,
        ii, jj;

    // double *temp;
    // double *p;

    /* get total number of processor rows and columns */
    nprow = MPI_Comm_size(comm_row, &nprow);
    npcol = MPI_Comm_size(comm_col, &npcol);

    /*get myrow, mycol*/
    MPI_Comm_rank( comm_row, &myrow );
    MPI_Comm_rank( comm_col, &mycol );

    icurrow = 0;        icurcol = 0;
    ii = jj = 0;

    /* malloc temp space for summation */
    double *work1 = (double *)malloc(m_a[myrow] * n_a[mycol] * sizeof(double));
    double *work2 = (double *)malloc(m_b[myrow] * n_b[mycol] * sizeof(double));

    printf("myrow: %d, mycol: %d\n", myrow, mycol);
    for (kk = 0; kk < k; kk += iwrk)
    {
        printf("Beginning of a new kk iteraction. Current kk=%d\n", kk);
        iwrk = min( nb, m_b[icurrow]);
        iwrk = min( iwrk, n_a[icurcol]);
        /* pack current iwrk columns of A into work 1 */
        printf("Start copying portions of Matrices\n");
        if ( mycol == icurcol%(npcol+1) )
            printf("debugging dlacpy: M=%d, N=%d, LDA=%d, LDB=%d\n",
                    m_a[myrow], iwrk, lda, m_a[myrow]);
            dlacpy_("General", &m_a[myrow], &iwrk, &A(0, jj), &lda, work1, &m_a[myrow]);
        /* pack current iwrk rows of B into work 2 */
        if ( myrow == icurrow%(nprow+1) )
            dlacpy_("General", &iwrk, &n_b[mycol], &B(ii, 0), &ldb, work2, &iwrk);
        /* broadcast work1 and work 2 */
        printf("Start Broadcasting\n");
        printf("root: icurrow=%d, icurcol=%d\n", icurrow, icurcol);

        MPI_Bcast(work1, m_a[ myrow ]*iwrk, MPI_DOUBLE, icurcol%(npcol+1), comm_row);
        MPI_Bcast(work2, n_b[ mycol ]*iwrk, MPI_DOUBLE, icurrow%(nprow+1), comm_col);

        /* update local block */
        // matrix multiplication
        printf("Process %d %d ready to perform dgemm with m=%d, n=%d, k=%d, lda=%d, ldb=%d, ldc=%d\n",
            myrow, mycol, m_c[myrow], n_c[mycol], iwrk, m_b[myrow], m_b[myrow], ldc);
        // dgemm_("No transpose", "No transpose", &m_c[myrow], &n_c[mycol],
        //         &iwrk, 1, work1, &n_a[myrow], work2, &iwrk, &d_zero,
        //         c, &ldc);
        // dgemm_("No transpose", "No transpose", &m_c[myrow], &n_c[mycol], &iwrk,
        //         &d_one, work1, &m_b[myrow], work2, &m_b[myrow], &d_zero, c, &ldc);
        // perform matrix multiplication using dgemm_
        dgemm_("No transpose", "No transpose", &m_c[myrow], &n_c[mycol], &iwrk,
               &d_one, work1, &m_a[myrow], work2, &iwrk, &d_one, c, &ldc);
        // matrix_multiply(m_c[myrow], n_c[mycol], iwrk, work1, m_a[myrow], work2, iwrk, c, ldc);
        printf("Process %d %d finished dgemm\n", myrow, mycol);
        printf("current kk: %d\n", kk);
        printf("current iwrk: %d\n", iwrk);
        printf("k: %d\n", k);


        /* update icurcol, icurrow, ii, jj */
        ii += iwrk;
        jj += iwrk;
        if ( jj>=n_a[ icurcol ] ) {
            icurcol++;
            // jj = 0;
        };
        if ( ii>=m_b[ icurrow ] ) {
            icurrow++;
            // ii = 0;
        };
        printf("icurrow: %d, icurcol: %d\n", icurrow, icurcol);
        MPI_Barrier(MPI_COMM_WORLD);
    }
    free(work1);
    free(work2);
}

void RING_Bcast (double *buf, int count, MPI_Datatype type, int root , MPI_Comm comm)
{
    int me, np;
    MPI_Status status;

    MPI_Comm_rank( comm, &me ); MPI_Comm_size(comm, &np);
    if (me != root)
    {
        printf("Recieve from %d\n", (me-1+np)%np);
        MPI_Recv(buf, count, type, (me-1+np)%np, MPI_ANY_TAG, comm, &status);
    }

    if ( (me+1)%np != root )
    {
        printf("Send to %d\n", (me+1)%np);
        MPI_Send(buf, count, type, (me+1)%np, 0, comm);
    }
}

void matrix_multiply(int m, int n, int k, double* A, int lda, double* B, int ldb, double* C, int ldc) {
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            double sum = 0.0;
            for (int p = 0; p < k; ++p) {
                sum += A[i * lda + p] * B[p * ldb + j];
            }
            C[i * ldc + j] = sum;
        }
    }
}
