CXX=nvcc

CUDA_INSTALL_PATH=/usr/local/cuda
CFLAGS= -I. -I$(CUDA_INSTALL_PATH)/include `pkg-config --cflags opencv`
LDFLAGS= -L$(CUDA_INSTALL_PATH)/lib64 -lcudart -lcuda `pkg-config --libs opencv`

all:
	$(CXX) $(CFLAGS) -c main.cu -o main.o
	$(CXX) main.o -o filters $(LDFLAGS)

clean:
	rm -f *.o filters



