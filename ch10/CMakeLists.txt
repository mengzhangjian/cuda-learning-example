CMAKE_MINIMUM_REQUIRED(VERSION 3.1)

CUDA_ADD_EXECUTABLE(pagetest pagetest.cu)
SET_PROPERTY(TARGET pagetest PROPERTY FOLDER chapter10)

CUDA_ADD_EXECUTABLE(stream_single stream_single.cu)
SET_PROPERTY(TARGET stream_single PROPERTY FOLDER chapter10)

CUDA_ADD_EXECUTABLE(stream_two stream_two.cu)
SET_PROPERTY(TARGET stream_two PROPERTY FOLDER chapter10)

CUDA_ADD_EXECUTABLE(stream_corrent stream_corrent.cu)
SET_PROPERTY(TARGET stream_corrent PROPERTY FOLDER chapter10)
