## Compilation and execution guide
1. Please clone our repository on Expanse.
2. Depending on which implementations you want to run, please follow the guide below.

* For the serial program, run the following  
    cd parallel-mha/true_serial_code  
    module purge  
    module load slurm  
    module load cpu/0.15.4    
    module load gpu/0.15.4  
    module load intel/17.0.7  
    g++ main.cpp multihead_attention.cpp matrix.cpp -o multihead_attention  
    ./multihead_attention 

* For the OpenMP implementation, run the following:
    cd parallel-mha/true_serial_code  
    module purge  
    module load slurm  
    module load cpu/0.15.4      
    module load gpu/0.15.4  
    module load intel/17.0.7  
    g++ -fopenmp main.cpp multihead_attention.cpp matrix.cpp -o multihead_attention  
    ./multihead_attention  

* For the OpenMP + MPI hybrid implementation, run the following:
    cd parallel-mha/hybrid_code 
    module purge  
    module load slurm  
    module load cpu/0.15.4    
    module load intel/19.1.1.217  
    module load openmpi/4.0.4  
    export I_MPI_CC=icc 
    mpicc -o summa summa.c  
    mpicxx -fopenmp main.cpp multihead_attention.cpp matrix.cpp -o multihead_attention
    export UCX_LOG_LEVEL=ERROR  
    mpirun -np 1 multihead_attention

* To run the summa program alone, please run the following:  
    cd parallel-mha/mpi  
    module purge  
    module load slurm  
    module load cpu/0.17.3b   
    module load aocc/3.2.0/io3s466  
    module load openmpi/4.1.3/xigazqd  
    export I_MPI_CC=icc  
    export OMP_NUM_THREADS=8  
    mpicc -qopenmp -o main_hybrid main_hybrid.c  
    mpirun -x I_MPI_PIN_DOMAIN=omp:compact -n 4 ./main_hybrid 2560 2560 2560 2560 2 2  