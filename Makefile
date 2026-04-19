CXX = g++
NVCC = nvcc
CUDA_INC = /usr/local/cuda/include
CUDA_LIB = /usr/local/cuda/lib64

CXXFLAGS = -Wall -std=c++20 -Iinclude -Isrc -I$(CUDA_INC)
NVCCFLAGS = -std=c++17 -Iinclude -Isrc -I$(CUDA_INC)

LDFLAGS = -L$(CUDA_LIB) -lcudart -lcurand

SRC_DIR := src
BUILD_DIR := build
TARGET := main

rwildcard = $(foreach d,$(wildcard $1/*),$(call rwildcard,$d,$2) $(filter $(subst *,%,$2),$d))

SRCS_CPP := $(call rwildcard,$(SRC_DIR),*.cpp)
SRCS_CU := $(call rwildcard,$(SRC_DIR),*.cu)

OBJS_CPP := $(patsubst $(SRC_DIR)/%.cpp,$(BUILD_DIR)/%.o,$(SRCS_CPP))
OBJS_CU := $(patsubst $(SRC_DIR)/%.cu,$(BUILD_DIR)/%.cu.o,$(SRCS_CU))

DEPS_CPP := $(OBJS_CPP:.o=.d)
DEPS_CU := $(OBJS_CU:.o=.d)
DEPS := $(DEPS_CPP) $(DEPS_CU)

OBJS := $(OBJS_CPP) $(OBJS_CU)

all: $(TARGET)

$(TARGET): $(OBJS)
	@$(CXX) -o $@ $^ $(LDFLAGS) $(LDLIBS)
	@echo ""
	@echo "Generation de l'executable $(TARGET) Done."

$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cpp
	@mkdir -p $(dir $@)
	@$(CXX) $(CXXFLAGS) -MMD -MP -c $< -o $@
	@echo "Compiltion $(CXX) $< Done." #changement pour faire en sorte de limite l'affichage sinon pas pratique dans la console

$(BUILD_DIR)/%.cu.o: $(SRC_DIR)/%.cu
	@mkdir -p $(dir $@)
	@$(NVCC) $(NVCCFLAGS) -MMD -MP -c $< -o $@
	@echo "Compiltion $(NVCC) $< Done."

clean:
	rm -rf $(BUILD_DIR) $(TARGET)

run: $(TARGET)
	@echo
	@echo Execution $(TARGET) ...
	@echo
	@./$(TARGET)
	@echo
	@echo Execution $(TARGET) Termine.

r: run
c: clean

-include $(DEPS)