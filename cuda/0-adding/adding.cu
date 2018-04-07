# include <stdlib.h> //needed for rand()
# include <stdio.h> //needed for printf()



//============================= GPU Kernel ====================================
__global__ void big_add(int *a, int *b, int *c, unsigned int N){
    // init thread id
    int tid;
    tid = blockIdx.x * blockDim.x + threadIdx.x;
    // stride is for big arrays, i.e. bigger than threads we have
    int stride = blockDim.x * gridDim.x;

    // do the operations
    while(tid < N){
        c[tid] = a[tid] + b[tid];
        tid += stride;
    }
}


//============================= CPU ===========================================
void cpu_add(int *a, int *b, int *c, unsigned int N){
    for(unsigned int i = 0; i < N; i++){
        c[i] = a[i] + b[i];
    }
}


//============================= MAIN ==========================================
int main(){

    // size of array to be added
    unsigned int N = 1000000;

    // DEVICE PROPERTIES
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop,0); // this allows us to query the device
                                      // if there are multiple devices, you can
                                      // loop thru them by changing the 0

    // print a couple of many properties
    printf("Max Threads per block: %d\n",prop.maxThreadsPerBlock);
    printf("Max Grid Size: %d x %d x %d\n",prop.maxGridSize[0],
           prop.maxGridSize[1], prop.maxGridSize[2]);

    // do a rough estimate of memory needed in stack
    printf("RAM needed estimate: %lu Mbytes\n", sizeof(int)*N*6/1000000);

    // allocate memory. Must use malloc() when dealing with arrays this large
    int *a = (int *) malloc(N* sizeof(int));
    int *b = (int *) malloc(N* sizeof(int));
    int *c = (int *) malloc(N* sizeof(int));
    int *d_a, *d_b, *d_c;

    // set up random number generator
    time_t tictoc;
    srand((unsigned) time(&tictoc));

    // set up timer
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // run the kernel
    unsigned int numBlocks, numThreads;
    numThreads = 1024; // max threads per block for compute capability > 2.0
    numBlocks = (N + numThreads - 1)/numThreads; // Round up

    // we don't want to create a grid larger than allowed, so make sure we're
    // not trying to do that
    if(numBlocks > prop.maxGridSize[1]){
        numBlocks = prop.maxGridSize[1];
    }

    // number of iterations to run the adds
    int iterations = 1000;

    // timer variable
    float milliseconds = 0;

    // open a file for logging
    FILE *f = fopen("gpu_add_times.txt", "w");

    // error handling for file open
    if (f == NULL)
    {
        printf("Error opening file!\n");
        exit(1);
    }
    printf("\n!!!\nRunning %d iterations of add on CPU and GPU\n", iterations);
    printf("Saving execution times to gpu_add_times.txt and cpu_add_times.txt\n");

    // loop through adds number of iterations
    for(int j = 0; j<iterations; j++) {
      // create random vectors
    	for(unsigned int i = 0; i < N; i++){
       	    a[i] = rand() % 100; // less than 100 for each element
       	    b[i] = rand() % 100;
    	}

        printf("GPU Iteration %d of %d...\n",j,iterations);

        // allocate memory on GPU
        cudaMalloc((void **) &d_a, sizeof(int) * N);
        cudaMalloc((void **) &d_b, sizeof(int) * N);
        cudaMalloc((void **) &d_c, sizeof(int) * N);

        // start timer
        cudaEventRecord(start);

        // copy memory from CPU to GPU
        cudaMemcpy(d_a, a, N * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_b, b, N * sizeof(int), cudaMemcpyHostToDevice);

        // call the kernel
        big_add <<< numBlocks, numThreads >>> (d_a, d_b, d_c, N);

        // stop the timer
        cudaEventRecord(stop);

        // copy the result from the device back to the host
        cudaMemcpy(c, d_c, N * sizeof(int), cudaMemcpyDeviceToHost);

        // sync the timers
        cudaEventSynchronize(stop);

        // free memory on device
        cudaFree(d_a);
        cudaFree(d_b);
        cudaFree(d_c);

        // calc elapsed time and print to file
        cudaEventElapsedTime(&milliseconds, start, stop);
        fprintf(f, "%f \n", milliseconds);
        printf("GPU Time: %f ms\n", milliseconds)
    }
    fclose(f); // close the file

    // open new file for the CPU
    FILE *g = fopen("cpu_add_times.txt", "w");
    if (f == NULL)
    {
        printf("Error opening file!\n");
        exit(1);
    }
    // create random vector
    for(int j = 0; j < iterations; j++) {
        for(unsigned int i = 0; i < N; i++){
       	    a[i] = rand() % 100;
       	    b[i] = rand() % 100;
    	}

	  printf("CPU Iteration %d of %d...\n",j,iterations);
        cudaEventRecord(start); // start timer
        cpu_add(a, b, c, N); // add the vectors
        cudaEventRecord(stop); // stop the timer
        cudaEventSynchronize(stop); // sync
        cudaEventElapsedTime(&milliseconds, start, stop); // calc time
        fprintf(g,"%f\n", milliseconds); // print to file
        printf("CPU Time: %f ms\n", milliseconds)
    }
    printf("Done!\n");
    fclose(g); // close the file
    return 0;
}
