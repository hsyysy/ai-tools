CXX = g++
CXXFLAGS = -std=c++17 -Wall -O3
LIBS = -lonnxruntime

TARGET = onnxinfo
SOURCE = onnxinfo.cpp

all: $(TARGET)

$(TARGET): $(SOURCE)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(SOURCE) $(LIBS)

clean:
	rm -f $(TARGET)

.PHONY: all clean
