#include "mpi.h"
/*We are using column major indexing*/
#define A( i,j ) (a[ j*lda+i ])
#define B( i,j ) (b[ j*ldb+i ])
#define C( i,j ) (c[ j*ldc+i ])

#define min( x,y ) ( (x) < (y) ? (x) : (y) )

int i_one = 1 /* used for constant passed to blas call */
double d_one=1.0,
        d_zero=0.0; /* used for constant passed to blas call */

void summa( m, n, k, nb,
            a, lda, b, ldb, c, ldc,
            m_a, n_a, m_b, n_b, m_c, n_c,
            comm_row, comm_col, work1, work2 )
int m, n, k,
    nb,           /* panel width */
    m_a[], n_a[], /* dimensions of blocks of A */
    m_b[], n_b[], /* dimensions of blocks of B */
    m_c[], n_c[], /* dimensions of blocks of C */
    lda, ldb, ldc;

double *a, *b, *c,
        *work1, *work2;

MPI_Comm comm_row, comm_col;
{
    int myrow, mycol,
        nprow, npcol,
        i, j, kk, iwrk,
        icurrow, icurcol,
        ii, jj;

    // double *temp;
    // double *p;

    // create 2d processor array if master task

    /*get myrow, mycol*/
    MPI_Comm_rank( comm_row, &myrow );
    MPI_Comm_rank( comm_col, &mycol );

    icurrow = 0;        icurcol = 0;
    ii = jj = 0;

    /* malloc temp space for summation */
    double *work1 = (double *) malloc( m_a[myrow]*n_a[mycol]*sizeof(double) )
    double *work2 = (double *) malloc( m_b[myrow]*n_b[mycol]*sizeof(double) )

    for ( kk=0; kk<k; kk+=iwrk ) {
        iwrk = min( nb, m_b[icurrow]-ii );
        iwrk = min( iwrk, n_a[icurcol]-jj );
        /* pack current iwrk columns of A into work 1 */
        if ( mycol == icurcol )
            dlacpy_("General", &m_a[myrow], &iwrk, &A(jj, 0), &lda, work1, &m_a[myrow]);
        /* pack current iwrk rows of B into work 2 */
        if ( myrow == icurrow )
            dlacpy_("General", &iwrk, &n_b[mycol], &B(0, ii), &ldb, work2, &iwrk);
        /* broadcast work1 and work 2 */
        RING_Bcast(work1, m_a[ myrow ]*iwrk, MPI_DOUBLE, icurcol, comm_row);
        RING_Bcast(work2, n_b[ mycol ]*iwrk, MPI_DOUBLE, icurrow, comm_col);

        /* update local block */
        // matrix multiplication
        dgemm_("No transpose", "No transpose", &m_c[myrow], &n_c[mycol],
                &iwrk, 1, work1, &m_b[myrow], work2, &iwrk, &d_one,
                c, &ldc);

        /* update icurcol, icurrow, ii, jj */
        ii += iwrk;
        jj += iwrk;
        if ( jj>=n_a[ icurcol ] ) { icurcol++; jj=0 };
        if ( ii>=m_b[ icurrow ] ) { icurrow++; ii=0 };
    }
    // free (temp);
}

RING_Bcast (double *buf, int count, MPI_Datatype type, int root , MPI_Comm comm)
{
    int me, np;
    MPI_Status status;

    MPI_Comm_rank( comm, me ); MPI_Comm_size(comm, np);
    if (me != root)
        MPI_Recv(buf, count, type, (me-1+np)%np, MPI_ANY_TAG, comm);
    if ( (me+1)%np != root )
        MPI_Send(buf, count, type, (me+1)%np, 0, comm);
}
