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
    mpirun -np 1 multihead_attention