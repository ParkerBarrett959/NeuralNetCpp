############################################################
#                 CMake Build of NeuralNetCpp              #
############################################################

# Project Setup
cmake_minimum_required(VERSION 3.22.0)
project(NeuralNetCpp)
set(CMAKE_CXX_STANDARD 23)

# Set include directories
include_directories(${PROJECT_SOURCE_DIR}/include)

# Set the core neural network source files
file(GLOB SOURCES "${PROJECT_SOURCE_DIR}/src/*.cc")

# Build the neural network static library
add_library(NeuralNetworkCpp STATIC ${SOURCES})

# Build the example executable
add_executable(trainMNist ${PROJECT_SOURCE_DIR}/example/TrainMNist.cc)

# Link to training library
target_link_libraries(trainMNist NeuralNetworkCpp)