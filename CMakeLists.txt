cmake_minimum_required(VERSION 3.1)
project(cuda_by_example)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

#folderf
set_property(GLOBAL PROPERTY USE_FOLDERS ON)

#outpath
set(EXECUTABLE_OUTPUT_PATH "${PROJECT_BINARY_DIR}/bin")


#Find OpenCV

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
link_directories(${OpenCV_LIBRARY_DIRS})
if(NOT OpenCV_FOUND)
    message(ERROR "OpenCV not found")
endif(NOT OpenCV_FOUND)

#CUDA
find_package(CUDA REQUIRED)

if(CUDA_FOUND)
#    add_subdirectory(ch04)
#    add_subdirectory(ch05)
    add_subdirectory(ch06)
else()
    message("CUDA not found!")
endif()
#FIND_PACKAGE(CUDA REQUIRED)
#
#set(CMAKE_CXX_STANDARD 11)
#INCLUDE_DIRECTORIES(${CUDA_INCLUDE_DIRS})
#include_directories(./common)
#
#
#
#cuda_add_executable(cuda main.cu)
#target_link_libraries(cuda /usr/local/cuda/lib64)
#target_link_libraries(cuda lib)