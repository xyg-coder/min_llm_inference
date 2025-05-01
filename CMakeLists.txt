cmake_minimum_required(VERSION 3.18)
project(min_llm_inference LANGUAGES CXX CUDA)

# Set the C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Enable CUDA
enable_language(CUDA)
find_package(CUDA REQUIRED)

# Add source files
set(INCLUDE_DIR ${CMAKE_SOURCE_DIR}/include)
set(SRC_DIR ${CMAKE_SOURCE_DIR}/src)
set(KERNELS_DIR ${SRC_DIR}/kernels)
set(TESTS_DIR ${CMAKE_SOURCE_DIR}/tests)

# Glob (recursively) all .cpp, .cu, .cuh, .hpp, .h files in include and src
file(GLOB_RECURSE ALL_HEADERS
    "${INCLUDE_DIR}/*.hpp"
    "${INCLUDE_DIR}/*.h"
    "${INCLUDE_DIR}/*.cuh"
)
file(GLOB_RECURSE ALL_SOURCES
    "${SRC_DIR}/*.cpp"
    "${SRC_DIR}/*.cu"
)

# Add the main library with all headers and source files
add_library(min_llm_inference
    ${ALL_HEADERS}
    ${ALL_SOURCES}
)

# Set CUDA properties
set_target_properties(min_llm_inference PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
)

# Include directories
target_include_directories(min_llm_inference 
    PUBLIC 
        ${INCLUDE_DIR}
        ${CUDA_INCLUDE_DIRS}
)

# Add Google Test
include(FetchContent)
FetchContent_Declare(
    googletest
    URL https://github.com/google/googletest/archive/refs/tags/release-1.12.1.zip
)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

# Add test executable
add_executable(gemm_test
    ${TESTS_DIR}/gemm_test.cpp
)

# Link libraries
target_link_libraries(gemm_test
    min_llm_inference
    gtest_main
)

# Enable testing
enable_testing()
add_test(NAME gemm_test COMMAND gemm_test)
