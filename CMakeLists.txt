cmake_minimum_required(VERSION 3.4)
project(3point_FD LANGUAGES CXX)

# find_package(IntelDPCPP REQUIRED)
set(CMAKE_CXX_COMPILER "dpcpp")
# To avoid naming device policy objects explicitly, add the –fsycl-unnamed-lambda option.
SET(CMAKE_CXX_FLAGS "-fsycl -fsycl-unnamed-lambda")
add_executable(${PROJECT_NAME} 3point_FD.cpp)
# dpcpp [–fsycl-unnamed-lambda] test.cpp [-ltbb] -o test -> example of a command line used to compile code that contains oneDPL parallel algorithms on Linux
# https://docs.oneapi.io/versions/latest/onedpl/pstl/dpcpp_policies_usage.html
target_link_libraries(${PROJECT_NAME} sycl tbb)
