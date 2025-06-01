cmake_minimum_required(VERSION 3.18)
project(min_llm_inference LANGUAGES CXX CUDA)

# uncomment when debugging
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g")
set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG} -g -G")

# Set the C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Enable CUDA
enable_language(CUDA)
find_package(CUDAToolkit REQUIRED)

# Add source files
set(INCLUDE_DIR ${CMAKE_SOURCE_DIR}/include)
set(SRC_DIR ${CMAKE_SOURCE_DIR}/src)
set(KERNELS_DIR ${SRC_DIR}/kernels)
set(TESTS_DIR ${CMAKE_SOURCE_DIR}/tests)

option(USE_ASYNC_ALLOC "If ON, default to async allocation" OFF)

if(USE_ASYNC_ALLOC)
    add_definitions(-DDEFAULT_ALLOC_METHOD=1)
else()
    add_definitions(-DDEFAULT_ALLOC_METHOD=0)
endif()

message("USE_ASYNC_ALLOC == ${USE_ASYNC_ALLOC}")

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
        ${CUDAToolkit_INCLUDE_DIRS}
)

# Add Google Test
include(FetchContent)
FetchContent_Declare(
    googletest
    URL https://github.com/google/googletest/archive/refs/tags/release-1.12.1.zip
)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

file(GLOB TEST_SOURCES "${TESTS_DIR}/*_test.cpp")

foreach(TEST_SRC ${TEST_SOURCES})
    # extract just the filename without extension (e.g. feed_forward_test)
    get_filename_component(TEST_NAME ${TEST_SRC} NAME_WE)

    add_executable(${TEST_NAME} ${TEST_SRC} ${TESTS_DIR}/test_utils.cpp)

    target_link_libraries(${TEST_NAME}
        min_llm_inference
        gtest_main
    )

    # Add tests/include directory as *PRIVATE* to each test target
    target_include_directories(${TEST_NAME} PRIVATE ${TESTS_DIR}/include)

    add_test(NAME ${TEST_NAME} COMMAND ${TEST_NAME})
endforeach()

enable_testing()
