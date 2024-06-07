/******************************************************************************
* FILE: summa_mpi.cpp
* DESCRIPTION:
*   SUMMA Matrix Multiply - MPI C++ Version for Arbitrary Matrix Sizes
*   Master initializes full matrices and distributes blocks for SUMMA computation.
*   Timers are used to measure and report performance metrics.
* AUTHOR: Billy Li, Zihan Zhao
* LAST REVISED: May 30, 2024
******************************************************************************/

#include <mpi.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>

#define MASTER 0

void SUMMA(MPI_Comm ROW_COMM, MPI_Comm COL_COMM, int Sr, int Sc,
           int ntrow, int ntcol, int lda, int ldb, int ldc,
           int n, int m, int p, float* locA, float* locB, float* Cloc, int bs)
{
    int c, r, id_col, id_row,
        offsetA, offsetB;

    float *A_tmp, *B_tmp;

    int n_size = n / Sr;
    int p_size = p / Sc;

    int blk_num = lcm(Sr, Sc);
    int m_size = m / blk_num;

    int blk_size_a = n_size * m_size;
    int blk_size_b = m_size * p_size;

    A_tmp = (float*) malloc(sizeof(float) * blk_size_a);
    B_tmp = (float*) malloc(sizeof(float) * blk_size_b);

    MPI_Comm_rank(ROW_COMM, &id_row);
    MPI_Comm_rank(COL_COMM, &id_col);

    offsetA = offsetB = 0;

    for(int k = 0; k < blk_num; k++)
    {
        r = k % Sc;
        c = k % Sr;

        if(id_row == r)
        {
            cp_matrix(&locA[offsetA], A_tmp, n_size, m_size, lda, m_size);
            offsetA += m_size;
        }

        if(id_col == c)
        {
            cp_matrix(&locB[offsetB], B_tmp, m_size, p_size, ldb, p_size);
            offsetB += (m_size * ldb);
        }

        MPI_Bcast(A_tmp, blk_size_a, MPI_FLOAT, r, ROW_COMM);
        MPI_Bcast(B_tmp, blk_size_b, MPI_FLOAT, c, COL_COMM);

        matmat_threads(ntrow, ntcol, m_size, p_size, ldc,
                       n_size, m_size, p_size, A_tmp, B_tmp, Cloc, bs);
    }

    free(A_tmp);
    free(B_tmp);
}
