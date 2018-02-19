# include <stdlib.h> //needed for rand()
# include <stdio.h> //needed for printf()

__global__ void big_add(int *a, int *b, int *c, unsigned int N){
    int tid;
    tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    while(tid < N){
        c[tid] = a[tid] + b[tid];
        tid += stride;
    }
}

void cpu_add(int *a, int *b, int *c, unsigned int N){
    for(unsigned int i = 0; i < N; i++){
        c[i] = a[i] + b[i];
    }
}

int main(){

    unsigned int N = 1000000;

    // DEVICE PROPERTIES
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop,0);
    printf("Max Threads per block: %d\n",prop.maxThreadsPerBlock);
    printf("Max Grid Size: %d x %d x %d\n",prop.maxGridSize[0],
           prop.maxGridSize[1], prop.maxGridSize[2]);

    printf("RAM needed estimate: %lu Mbytes\n", sizeof(int)*N*6/1000000);

    int *a = (int *) malloc(N* sizeof(int));
    int *b = (int *) malloc(N* sizeof(int));
    int *c = (int *) malloc(N* sizeof(int));
    int *d_a, *d_b, *d_c;

    // set up random number generator
    time_t tictoc;
    srand((unsigned) time(&tictoc));


    printf("copying memory...\n");

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // run the kernel
    unsigned int numBlocks, numThreads;
    numThreads = 1024;
    printf("calculating numBlocks...\n");
    numBlocks = (N + numThreads - 1)/numThreads;

    if(numBlocks > prop.maxGridSize[1]){
        numBlocks = prop.maxGridSize[1];
    }

    int iterations = 1000;
    float milliseconds = 0;

    FILE *f = fopen("gpu_add_times.txt", "w");
    if (f == NULL)
    {
        printf("Error opening file!\n");
        exit(1);
    }

    for(int j = 0; j<iterations; j++) {

    	for(unsigned int i = 0; i < N; i++){
       	    a[i] = rand() % 100;
       	    b[i] = rand() % 100;
    	}

 

        printf("GPU Iteration %d of %d...\n",j,iterations);
        // allocate memory
        cudaMalloc((void **) &d_a, sizeof(int) * N);
        cudaMalloc((void **) &d_b, sizeof(int) * N);
        cudaMalloc((void **) &d_c, sizeof(int) * N);

        cudaEventRecord(start);
        // copy memory

        cudaMemcpy(d_a, a, N * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_b, b, N * sizeof(int), cudaMemcpyHostToDevice);

        big_add << < numBlocks, numThreads >> > (d_a, d_b, d_c, N);

        cudaEventRecord(stop);

        cudaMemcpy(c, d_c, N * sizeof(int), cudaMemcpyDeviceToHost);

        cudaEventSynchronize(stop);

        cudaFree(d_a);
        cudaFree(d_b);
        cudaFree(d_c);

        cudaEventElapsedTime(&milliseconds, start, stop);
        fprintf(f, "%f \n", milliseconds);
    }
    fclose(f);

    FILE *g = fopen("cpu_add_times.txt", "w");
    if (f == NULL)
    {
        printf("Error opening file!\n");
        exit(1);
    }
    for(int j = 0; j < iterations; j++) {
        for(unsigned int i = 0; i < N; i++){
       	    a[i] = rand() % 100;
       	    b[i] = rand() % 100;
    	}
	printf("CPU Iteration %d of %d...\n",j,iterations);
        cudaEventRecord(start);
        cpu_add(a, b, c, N);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&milliseconds, start, stop);
        fprintf(g,"%f\n", milliseconds);
    }
    fclose(g);
    return 0;
}
