# =========================================================
# Makefile for TensorEngine
#
# Layout expected:
#   source/main.cpp
#   source/utils.cpp
#   source/arthematic.cpp
#   include/*.hpp
#   __cuda__/source/kernel.cu
#   __cuda__/include/kernel.cuh
#
# Usage:
#   make            build everything -> bin/tensor_engine
#   make run        build and run
#   make clean      remove build artifacts
#   make DEBUG=1    build with debug symbols, no optimization
# =========================================================

# ---------- Toolchain ----------
CXX      := g++-15
NVCC     := nvcc

# ---------- Directories ----------
SRC_DIR      := source
INC_DIR      := include
CUDA_SRC_DIR := __cuda__/source
CUDA_INC_DIR := __cuda__/include
BUILD_DIR    := build
BIN_DIR      := bin

TARGET := $(BIN_DIR)/tensor_engine

# ---------- Sources ----------
CPP_SOURCES  := $(wildcard $(SRC_DIR)/*.cpp)
CUDA_SOURCES := $(wildcard $(CUDA_SRC_DIR)/*.cu)

CPP_OBJECTS  := $(patsubst $(SRC_DIR)/%.cpp,$(BUILD_DIR)/%.o,$(CPP_SOURCES))
CUDA_OBJECTS := $(patsubst $(CUDA_SRC_DIR)/%.cu,$(BUILD_DIR)/%.o,$(CUDA_SOURCES))

OBJECTS := $(CPP_OBJECTS) $(CUDA_OBJECTS)

# ---------- Flags ----------
STD          := -std=c++17
INCLUDES     := -I$(INC_DIR) -I$(CUDA_INC_DIR)

ifdef DEBUG
    CXXFLAGS  := $(STD) $(INCLUDES) -g -O0 -Wall -Wextra
    NVCCFLAGS := $(STD) $(INCLUDES) -g -G -O0
else
    CXXFLAGS  := $(STD) $(INCLUDES) -O2 -Wall -Wextra
    NVCCFLAGS := $(STD) $(INCLUDES) -O2
endif

# nvcc drives the final link so CUDA runtime libs are pulled in correctly
LDFLAGS :=

# ---------- Rules ----------
.PHONY: all run clean dirs

all: dirs $(TARGET)

dirs:
	@mkdir -p $(BUILD_DIR) $(BIN_DIR)

# Link: use nvcc so it adds -lcudart / cuda lib paths automatically
$(TARGET): $(OBJECTS)
	$(NVCC) $(STD) $(OBJECTS) -o $@ $(LDFLAGS)
	@echo "Built $@"

# Compile C++ sources with g++
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cpp | dirs
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Compile CUDA sources with nvcc
$(BUILD_DIR)/%.o: $(CUDA_SRC_DIR)/%.cu | dirs
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

run: all
	./$(TARGET)

clean:
	rm -rf $(BUILD_DIR) $(BIN_DIR)