CMAKE_MINIMUM_REQUIRED(VERSION 3.1)

CUDA_ADD_EXECUTABLE(histogram histogram.cu)
SET_PROPERTY(TARGET histogram PROPERTY FOLDER chapter09)
TARGET_LINK_LIBRARIES(histogram ${OpenCV_LIBS})