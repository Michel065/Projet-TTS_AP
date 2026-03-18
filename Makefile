CXX = g++
CXXFLAGS = -Wall -std=c++20 -Iinclude -Isrc -MMD -MP -Iext_inc
LDFLAGS =

SRC_DIR := src
BUILD_DIR := build
TARGET := main.exe

rwildcard = $(foreach d,$(wildcard $1/*),$(call rwildcard,$d,$2) $(filter $(subst *,%,$2),$d))

SRCS := $(call rwildcard,$(SRC_DIR),*.cpp)
OBJS := $(patsubst $(SRC_DIR)/%.cpp,$(BUILD_DIR)/%.o,$(SRCS))
DEPS := $(OBJS:.o=.d)

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CXX) -o $@ $^ $(LDFLAGS)

$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cpp
	@if not exist "$(subst /,\,$(dir $@))" mkdir "$(subst /,\,$(dir $@))"
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	if exist "$(BUILD_DIR)" rmdir /s /q "$(BUILD_DIR)"
	if exist "$(TARGET)" del /q "$(TARGET)"

run: $(TARGET)
	@echo.
	@echo Execution $(TARGET) ... 
	@echo.
	@.\$(TARGET)
	@echo.
	@echo Execution $(TARGET) Termine.


r: run
c: clean
	
-include $(DEPS)