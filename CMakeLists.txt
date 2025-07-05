############################################################
#                 CMake Build of NeuralNetCpp              #
############################################################

# Project Setup
cmake_minimum_required(VERSION 3.22.0)
project(NeuralNetCpp)
set(CMAKE_CXX_STANDARD 23)

# Set include directories
include_directories(${PROJECT_SOURCE_DIR}/include)

# Set the core navigation source files
file(GLOB SOURCES "${PROJECT_SOURCE_DIR}/src/*.cc")

# Build the neural network static library
add_library(NeuralNetwork STATIC ${SOURCES})