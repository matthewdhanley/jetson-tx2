# Getting Started with CUDA
Comput Unified Device Architecture (CUDA) by NVIDIA allows a user to write programs in C and compile using ncc to make use of a CUDA-enabled GPU device. This enables a GPU device to run single instruction, multiple thread (SIMT) code. This gives the ability for creating highly parallized programs.

## The Thread
The thread is like a little computer. Threads in a CUDA program have their own program counters and registers and are run simutaneousluy with multiple different threads within the same _block_. All threads are able to share a "global memory," but also have their own "privite memory."
![Thread](https://github.com/matthewdhanley/jetson-tx2/blob/master/cuda/intro/img/thread.png)
## The Block
A block is made up of many threads. Within a block, threads share a very fast "shared memory" as well as instruction stream. These instructions run in parallel. A block on the Jetson TX2 can have up to 1024 threads. Blocks can have many dimensions (1, 2, or 3) as specified by the user. All blocks within a _grid_ must be the same size.
![Thread](https://github.com/matthewdhanley/jetson-tx2/blob/master/cuda/intro/img/block.png)
## The Grid
A grid is a two-dimensional collection of blocks. The dimensions are specified by the user.
![Thread](https://github.com/matthewdhanley/jetson-tx2/blob/master/cuda/intro/img/grid.png)
## Finding the Thread
At the start of a program, threads identify themselves. This helps map where a thread is within a block within a grid. The table below shows how to index into grids.
![Thread](https://github.com/matthewdhanley/jetson-tx2/blob/master/cuda/intro/img/full.png)

| Variable | Description |
| --- | --- |
| gridDim.(x,y,z) | Gives the dimensions of a grid |
| blockDim.(x,y,z) | Gives the dimension of the blocks |
| blockIdx.(x,y,z) | Gives the location of the block within a grid |
| threadIdx.(x,y,z) | Gives the location of a thread within its block |

## Kernels
A _kernel_ is the "unit of work" that is sent to the GPU by the CPU. A kernel is specified by the `__global__` tag. A kenel is launched as follows:
```
my_gpu_fun<<<dimGrid, dimBlock>>>(var1, var2, var3);
```
`my_gpu_fun` is the kernel that is preceeded by the `__global__` tag. The angled brackets, `<<< >>>`, tell the GPU how many blocks within a grid (i.e. `gridDim`) and how many threads within a block (i.e. `blockDim`). `dimGrid` and `dimBlock` are both of type dim3. Finally, the parameters are passed to the kernel.

### References
[[1]](https://www.shodor.org/media/content/petascale/materials/UPModules/matrixMultiplication/moduleDocument.pdf) Robert Hochberg. _Matrix Multiplication with CUDA - A basic introduction to the CUDA programming model._ 2012.
