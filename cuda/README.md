# Getting Started with CUDA
Comput Unified Device Architecture (CUDA) by NVIDIA allows a user to write programs in C and compile using ncc to make use of a CUDA-enabled GPU device. This enables a GPU device to run single instruction, multiple thread (SIMT) code. This gives the ability for creating highly parallized programs.

## The Thread
The thread is like a little computer. Threads in a CUDA program have their own program counters and registers and are run simutaneousluy with multiple different threads within the same _block_. All threads are able to share a "global memory," but also have their own "privite memory."
![Thread](https://github.com/matthewdhanley/jetson-tx2/blob/master/cuda/intro/img/thread.png)
## The Block
A block is made up of many threads. Within a block, threads share a very fast "shared memory" as well as instruction stream. These instructions run in parallel. A block on the Jetson TX2 can have up to 1024 threads. Blocks can have many dimensions (1, 2, or 3) as specified by the user. All blocks within a _grid_ must be the same size.
![Block](https://github.com/matthewdhanley/jetson-tx2/blob/master/cuda/intro/img/block.png)
## The Grid
A grid is a two-dimensional collection of blocks. The dimensions are specified by the user.
![Grid](https://github.com/matthewdhanley/jetson-tx2/blob/master/cuda/intro/img/grid.PNG)
## Finding the Thread
At the start of a program, threads identify themselves. This helps map where a thread is within a block within a grid. The table below shows how to index into grids.
![Full](https://github.com/matthewdhanley/jetson-tx2/blob/master/cuda/intro/img/full.PNG)

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

## Optomization
Again, the whole reason for using a GPU is to speed up code. Optomizing your code is a great way to get even more gains. Writing code without thoughts of optomizing can make GPUs even slower than CPUs. Here are some basic optomization terms one should be familiar with.

## Occupancy
Ratio of active warps per SM to total number of possible active warps on the SM.

### Cycle
A cycle, also known as an _instruction cycle_ is the process of taking an instruction from the instruction register and performing the requested action. Examples of instructions include fetching (grabbing a value from a register), jumping to another location in the instruction stack, adding, subtracting, logical comparisons, loading memory, writing memory, and many many more. Each instruction can (and usualy does) take more than one cycle.

### Latency
Time required for an operation to complete. Arithmetic takes about 20 cycles whereas a memory access will take on the order of 400 cycles. 

### Throughput
Number of operations that complete per cycle. Throughput is a rate, latency is a time.

### Hiding Latency
One way of increasing the efficiency and thus speed is to _hide latency_. This basically just means that the computer performs other actions while it is waiting for another instruction to complete. For example, if a memory load is being done, some adds could be completed while the value is being fetched. On a high level, this can be done by making use of all registers avalible to a thread. 

### Work smarter, not harder with CUDA Runtime Functions
Using runtime functions, ```cudaOccupancyMaxActiveBlocksPerMultiprocessor```, ```cudaOccupancyMaxPotentialBlockSize```, and ```cudaOccupancyMaxPotentialBlockSizeVariableSMem```, one can easily calculate a block size that will maximize occupancy. This will help hide latency and improve speed of the code. ```VariableSmem``` can be used for kernels where shared memory size depends on block size (note that shared memory is usually definded using a constant value). The below code from Mark Harris shows how this can be done.
``` C
#include "stdio.h"

__global__ void MyKernel(int *array, int arrayCount) 
{ 
  int idx = threadIdx.x + blockIdx.x * blockDim.x; 
  if (idx < arrayCount) 
  { 
    array[idx] *= array[idx]; 
  } 
} 

void launchMyKernel(int *array, int arrayCount) 
{ 
  int blockSize;   // The launch configurator returned block size 
  int minGridSize; // The minimum grid size needed to achieve the 
                   // maximum occupancy for a full device launch 
  int gridSize;    // The actual grid size needed, based on input size 

  cudaOccupancyMaxPotentialBlockSize( &minGridSize, &blockSize, 
                                      MyKernel, 0, 0); 
  // Round up according to array size 
  gridSize = (arrayCount + blockSize - 1) / blockSize; 

  MyKernel<<< gridSize, blockSize >>>(array, arrayCount); 

  cudaDeviceSynchronize(); 

  // calculate theoretical occupancy
  int maxActiveBlocks;
  cudaOccupancyMaxActiveBlocksPerMultiprocessor( &maxActiveBlocks, 
                                                 MyKernel, blockSize, 
                                                 0);

  int device;
  cudaDeviceProp props;
  cudaGetDevice(&device);
  cudaGetDeviceProperties(&props, device);

  float occupancy = (maxActiveBlocks * blockSize / props.warpSize) / 
                    (float)(props.maxThreadsPerMultiProcessor / 
                            props.warpSize);

  printf("Launched blocks of size %d. Theoretical occupancy: %f\n", 
         blockSize, occupancy);
}
```
This section of code is a basic kernel. It takes an input array and squares each element.
``` C
__global__ void MyKernel(int *array, int arrayCount) 
{ 
  int idx = threadIdx.x + blockIdx.x * blockDim.x; 
  if (idx < arrayCount) 
  { 
    array[idx] *= array[idx]; 
  } 
} 
```
<hr>
### The Code, Explained

This is the beginning of the kernel launch function. It takes an array and number of elements in the array. This specific section initializes three variables, ```blocksize```, ```minGridSize```, and ```grid size```. The function of these variables exists in the comments.
``` C
void launchMyKernel(int *array, int arrayCount) 
{ 
  int blockSize;   // The launch configurator returned block size 
  int minGridSize; // The minimum grid size needed to achieve the 
                   // maximum occupancy for a full device launch 
  int gridSize;    // The actual grid size needed, based on input size 
```
<br>

This is the first instance of something new. The ```cudaOccupancyMaxPotentialBlockSize``` runtime function takes five inputs. The first is the address for the minimum grid size, the second is the address for the block size, the third is the kernel, the fourth is dynamic shared memory size (Per-block dynamic shared memory usage intended, in bytes), and the last is the block size limit (0 means no limit). The last two default to zero. This function returns (by reference) the minimum grid size and block size needed to theoretically get the best occupancy.
``` C
 cudaOccupancyMaxPotentialBlockSize( &minGridSize, &blockSize, 
                                      MyKernel, 0, 0); 
```
<br>

This rounds up the total number of blocks launch based on the size of the array.
``` C
  gridSize = (arrayCount + blockSize - 1) / blockSize; 
```
<br>

Launches the kernel and syncs the device.
``` C
  MyKernel<<< gridSize, blockSize >>>(array, arrayCount); 

  cudaDeviceSynchronize(); 
```
<br>

This function is used to calculate the maximum number of active blocks. It takes a pointer to the variable in which the function will return the occupancy. The next parameter is the kernel. Next is the block size used. Finally is the dynamic shared memory size (Per-block dynamic shared memory usage intended, in bytes).
``` C
  int maxActiveBlocks;
  cudaOccupancyMaxActiveBlocksPerMultiprocessor( &maxActiveBlocks, 
                                                 MyKernel, blockSize, 
                                                 0);
```
<br>

This section gets properties from the device. ```int device;``` initializes the variable which will hold the device properties. Assuming one device, ```cudaGetDevice(&device);``` will return the value corresponding to the device. ```props``` is a struct that will hold information pertaining to the device. ```cudaGetDeviceProperties``` queries the device and returns the stats into the ```props``` variable.
``` C
  int device;
  cudaDeviceProp props;
  cudaGetDevice(&device);
  cudaGetDeviceProperties(&props, device);
```

<br>
This calculates the occupancy achieved. The number of active threads divided by the warp size is divided by the theoretical max number of threads per multiprocessor (i.e. active/max). Finally it's printed to the user.
``` C
  float occupancy = (maxActiveBlocks * blockSize / props.warpSize) / 
                    (float)(props.maxThreadsPerMultiProcessor / 
                            props.warpSize);

  printf("Launched blocks of size %d. Theoretical occupancy: %f\n", 
         blockSize, occupancy);
}
```


### References
[[1]](https://www.shodor.org/media/content/petascale/materials/UPModules/matrixMultiplication/moduleDocument.pdf) Robert Hochberg. _Matrix Multiplication with CUDA - A basic introduction to the CUDA programming model._ 2012.
http://www.oocities.org/mc_introtocomputers/Simple_CPU_Instructions_with_registers.htm
https://devblogs.nvidia.com/cuda-pro-tip-occupancy-api-simplifies-launch-configuration/
https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__HIGHLEVEL.html#group__CUDART__HIGHLEVEL_1gee5334618ed4bb0871e4559a77643fc1
