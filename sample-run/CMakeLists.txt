cmake_minimum_required(VERSION 3.17)
project(KMeansClustering CUDA CXX)

set (CMAKE_CXX_STANDARD 11)
set(CMAKE_CUDA_STANDARD 14)

include_directories(../k-means-clustering ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

add_executable(main main.cu)

set_target_properties(
        main
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
