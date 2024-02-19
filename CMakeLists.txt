cmake_minimum_required(VERSION 3.14 FATAL_ERROR)
project(GNNBench)

# Set C++ standard
set(CMAKE_CXX_STANDARD 14)

# Download dependencies
include(FetchContent)

# Fast Matrix Market
FetchContent_Declare(
        fast_matrix_market
        GIT_REPOSITORY https://github.com/alugowski/fast_matrix_market
        GIT_TAG main
        GIT_SHALLOW TRUE
)
FetchContent_MakeAvailable(fast_matrix_market)

# Libtorch
set(Torch_DIR ${TORCH_INSTALL_DIR}/share/cmake/Torch)
Set(FETCHCONTENT_QUIET FALSE)
FetchContent_Declare(
  Torch
  URL           https://download.pytorch.org/libtorch/cpu/libtorch-macos-arm64-2.2.0.zip
  SOURCE_DIR    ${TORCH_INSTALL_DIR}
  OVERRIDE_FIND_PACKAGE
  DOWNLOAD_EXTRACT_TIMESTAMP
)
FetchContent_Populate(Torch)


find_package(Torch REQUIRED)

# Set flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

# Add source files
set(SOURCE_FILES
    src/main.cpp
    # Add more source files if needed
)

# Add the executable
add_executable(GNNBench ${SOURCE_FILES})

# Link against libtorch
target_link_libraries(GNNBench ${TORCH_LIBRARIES} fast_matrix_market::fast_matrix_market)

# Include libtorch headers
target_include_directories(GNNBench PRIVATE ${TORCH_INCLUDE_DIRS})
