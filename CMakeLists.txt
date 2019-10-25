cmake_minimum_required(VERSION 3.15)
project(HPC3_CUDA)

set(CMAKE_CXX_STANDARD 14)

add_executable(HPC3_CUDA voronoi_gpu.cpp)