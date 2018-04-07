CXX=g++

CUDA_INSTALL_PATH=/usr/local/cuda
CFLAGS= -I. -I$(CUDA_INSTALL_PATH)/include `pkg-config --cflags opencv`
LDFLAGS= -L$(CUDA_INSTALL_PATH)/lib64 -lcudart -lcuda `pkg-config --libs opencv`

#Uncomment the line below if you dont have CUDA enabled GPU
#EMU=-deviceemu

ifdef EMU
CUDAFLAGS+=-deviceemu
endif

all:
	$(CXX) $(CFLAGS) -c main.cpp -o main.o
	nvcc $(CUDAFLAGS) -c gpu_kernels.cu -o gpu_kernels.o
	$(CXX) main.o gpu_kernels.o -o cam $(LDFLAGS)

clean:
	rm -f *.o cam
