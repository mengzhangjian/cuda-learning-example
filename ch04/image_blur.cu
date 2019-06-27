#include "opencv2/opencv.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include "cuda.h"
#include <math.h>
#include "../common/image.h"

using namespace cv;
using namespace std;


__global__ void color_to_gray(unsigned char *dev_bitmap, unsigned char *host_out, int width, int height)
{
    int row = threadIdx.x + blockIdx.x * blockDim.x;
    int col = threadIdx.y + blockIdx.y * blockDim.y;
    if(col < width && row < height){

        int grayoffset = row * width + col;

        int rgboffset = grayoffset * 3;

        unsigned char r = dev_bitmap[rgboffset ];
        unsigned char g = dev_bitmap[rgboffset + 2];
        unsigned char b = dev_bitmap[rgboffset + 3];
        host_out[grayoffset] = 0.21f * r + 0.71f * g + 0.07f * b;
    }

}

int main(void){

    Mat image = imread("4.jpg");
    Mat out = Mat::zeros(image.cols, image.rows, CV_8UC1);
    unsigned char *dev_bitmap;
    unsigned char *host_out;
    cudaMalloc((void**)&dev_bitmap, image.rows * image.cols * 4);
    cudaMemcpy(dev_bitmap, image.data, image.rows * image.cols * 4, cudaMemcpyHostToDevice);

    cudaMalloc((void**)&host_out, image.rows * image.cols);

    dim3 BlocksPerGrid(ceil(image.cols / 16.0), ceil(image.rows / 16.0), 1);
    dim3 ThreadsPerBlock(16, 16, 1);
    color_to_gray<<<BlocksPerGrid, ThreadsPerBlock>>>(dev_bitmap, host_out, image.cols, image.rows);
    cudaMemcpy(out.data, host_out, image.rows * image.cols, cudaMemcpyDeviceToHost);
    imwrite("blur.jpg", out);
    cudaFree(dev_bitmap);
    cudaFree(host_out);
    return 0;

}