#include <stdio.h>
#include <stdlib.h>
#include <assert.h>


void native_mat_mult(int *a, int *b, int *result, int m, int n, int k)
{
	for (int i = 0; i < m; i++)
	{
		for (int j = 0; j < n; j++)
		{
			int tmp = 0.0;
			for (int h = 0; h < n; h++)
			{
				tmp += a[i * n + h] * b[h * k + j];
			}
		result[i * k + j] = tmp;
		}
	}

}


int main(int argc, char const *argv[])
{
	int m, n, k;
	printf("Input m n k::\n");
	scanf("%d %d %d", &m, &n, &k);

	int *cpu_a, *cpu_b, *cpu_result;

	cudaMallocHost((void **) &cpu_a, sizeof(int)*m*n);
	cudaMallocHost((void **) &cpu_b, sizeof(int)*m*n);
	cudaMallocHost((void **) &cpu_result, sizeof(int)*m*n);

	for (int i = 0; i < m; i++)
	{
		for (int j = 0; j < n; j++)
		{
			cpu_a[i * n + j] = rand () % 1024;
		}
	}

        for (int i = 0; i < n; i++)
        {
                for (int j = 0; j < k; j++)
                {
                        cpu_b[n * k + j] = rand () % 1024;
                }
        }

	float cpu_time;

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	//cpu
	cudaEventRecord(start,0);

	native_mat_mult(cpu_a, cpu_b, cpu_result, m, n, k);

	cudaEventRecord(stop,0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&cpu_time, start, stop);
	
	printf("Time on mat mult of %d%d * %d%d on CPU: %f ms.\n\n", m, n, n, k, cpu_time);

	cudaFreeHost(cpu_a);
	cudaFreeHost(cpu_b);
	cudaFreeHost(cpu_result);
	return 0;
}
