CXX = g++
CXXFLAGS = -Wall -std=c++17 -Iinclude -Isrc -MMD -MP

SRC_DIR = src
BUILD_DIR = build
TARGET = main.exe

SRCS = src/main.cpp
OBJS = build/main.o
DEPS = $(OBJS:.o=.d)

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CXX) -o $@ $^

build/main.o: src/main.cpp
	if not exist build mkdir build
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	if exist build rmdir /s /q build
	if exist $(TARGET) del /q $(TARGET)

run: $(TARGET)
	.\$(TARGET)

-include $(DEPS)