cmake_minimum_required(VERSION 3.4)
project(vec_reduction LANGUAGES CXX)

# find_package(IntelDPCPP REQUIRED)
set(CMAKE_CXX_COMPILER "dpcpp")
SET(CMAKE_CXX_FLAGS "-fsycl -fsycl-unnamed-lambda")
add_executable(${PROJECT_NAME} vec_reduction_sum.cpp)
