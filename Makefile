SRC_PATH=src
BUILD_DIR=build

LIB_PATH=$(CURDIR)/lib
GPUTK_LIB_PATH=$(LIB_PATH)/libgputk/lib
GPUTK_SRC_PATH=$(LIB_PATH)/libgputk
NVCC=nvcc
CXX=g++

CUDA_FLAGS=-I $(GPUTK_SRC_PATH)
CXX_FLAGS=-std=c++11 -I $(GPUTK_SRC_PATH)
LIBS=-lm -L $(GPUTK_LIB_PATH) -lgputk

SOURCES := $(wildcard $(SRC_PATH)/*cpu.cpp)
OBJECTS := $(SOURCES:%=$(BUILD_DIR)/%.o)
CUDA_SOURCES := $(wildcard $(SRC_PATH)/*.cu)
CUDA_OBJECTS := $(CUDA_SOURCES:%=$(BUILD_DIR)/%.o)

TEST_EXE=$(BUILD_DIR)/main

# Main
main: $(BUILD_DIR) $(TEST_EXE)

$(TEST_EXE): $(BUILD_DIR)/main.o $(OBJECTS) $(CUDA_OBJECTS)
	$(NVCC) $(CUDA_FLAGS) -o $@ $^ $(LIBS)

# Main object file
$(BUILD_DIR)/main.o: $(SRC_PATH)/main.cpp
	$(CXX) $(CXX_FLAGS) -c $< -o $@

# C++ files
$(BUILD_DIR)/%.cpp.o: %.cpp
	$(CXX) $(CXX_FLAGS) $(CUDA_FLAGS) -g -c $< -o $@

# CUDA files
$(BUILD_DIR)/%.cu.o: %.cu
	$(NVCC) $(CUDA_FLAGS) -lineinfo -c $< -o $@

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)/src

clean:
	rm -rf $(BUILD_DIR)

.SUFFIXES: .c .cu .o
.PHONY: all main clean
