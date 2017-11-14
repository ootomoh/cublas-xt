NVCC=nvcc
NVCCFLAGS= -arch=sm_60 -std=c++11
CXX=g++
CXXFLAGS=-std=c++11
OBJS=main.o
BIN=exec

$(BIN): $(OBJS)
	$(NVCC) $(NVCCFLAGS) $+ -o $@

.SUFFIXES: .o .cu .cpp

.cu.o:
	$(NVCC) $(NVCCFLAGS) $< -c -o $@

.cpp.o:
	$(CXX) $(CXXFLAGS) $< -c -o $@


clean:
	rm -rf *.o
	rm -rf $(BIN)
