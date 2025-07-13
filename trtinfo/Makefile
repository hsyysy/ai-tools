CXX = g++
CXXFLAGS = -std=c++17 -O3 -Wall -Wextra

CUDA_ROOT = /usr/local/cuda

INCLUDES = -I$(CUDA_ROOT)/include
LDFLAGS = -L$(CUDA_ROOT)/lib64
LIBS = -lnvinfer -lcudart

TARGET = trtinfo
SOURCE = trtinfo.cpp

$(TARGET): $(SOURCE)
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(SOURCE) -o $(TARGET) $(LDFLAGS) $(LIBS)

clean:
	rm -f $(TARGET)

.PHONY: clean
