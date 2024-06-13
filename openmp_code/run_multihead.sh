#!/bin/bash

# Define an array of thread counts
thread_counts=(2 4 6 8 16 32)

# Loop through each thread count
for num_threads in "${thread_counts[@]}"; do
    # Set the number of OpenMP threads
    export OMP_NUM_THREADS=$num_threads
    
    # Execute the program and redirect output to a file
    ./multihead_attention > "openmp_${num_threads}.txt"
    
    # Optional: Echo to know which step is executing
    echo "Ran with $num_threads threads, output in openmp_${num_threads}.txt"
done
