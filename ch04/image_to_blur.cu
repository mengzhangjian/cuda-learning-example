//
// Created by zhangjian on 19-6-30.
//

#include "opencv2/opencv.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <math.h>
#include "cuda.h"

using namespace cv;
using namespace std;
const int BLUR_SIZE = 3;
__global__ void blurkernel(unsigned char *dev_in, unsigned char *dev_out, int w, int h){

    int col = threadIdx.x + blockDim.x * blockIdx.x;
    int row = threadIdx.y + blockDim.y * blockIdx.y;

    int offset = row * w + col;
    if(col < w && row < h)
    {
        int pixVal_red = 0;
        int pixVal_green = 0;
        int pixVla_blue = 0;
        int pixels = 0;

        for(int blurRow = -BLUR_SIZE; blurRow < BLUR_SIZE + 1; blurRow++)
            for(int blurCol = -BLUR_SIZE; blurCol < BLUR_SIZE + 1; blurCol++)
            {
             int curRow = row + blurRow;
             int curCol = col + blurCol;
             if(curCol > -1 && curCol < w && curRow > -1 && curRow < h)
             {
                 int rgboffset = 3 * (curRow * w + curCol);
                 pixVal_red += dev_in[rgboffset];
                 pixVal_green += dev_in[rgboffset + 1];
                 pixVla_blue += dev_in[rgboffset + 2];
                 pixels++;
             }

            }
        dev_out[offset * 3] = (unsigned char)(pixVal_red/ pixels);
        dev_out[offset * 3 + 1] = (unsigned char)(pixVal_green / pixels);
        dev_out[offset * 3 + 2] = (unsigned char)(pixVla_blue / pixels);
    }
}

int main(void){

    Mat image = imread("2.jpg");
    Mat out = Mat::zeros(image.rows, image.cols, CV_8UC3);
    unsigned char *dev_bitmap;
    unsigned char *dev_out;
    float elapsedTime;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaMalloc( (void**)&dev_bitmap, image.rows * image.cols* 3 * sizeof(unsigned char));
    cudaMemcpy(dev_bitmap, image.data, image.rows * image.cols* 3 * sizeof(unsigned char), cudaMemcpyHostToDevice);
    cudaMalloc( (void**)&dev_out, image.rows * image.cols* 3 * sizeof(unsigned char));
    dim3 blocksPerGrid(ceil(image.rows / 16.0), ceil(image.cols / 16.0), 1);
    dim3 threadsPerBlock(16, 16, 1);
    cudaEventRecord(start, 0);
    blurkernel<<<blocksPerGrid, threadsPerBlock>>>(dev_bitmap, dev_out,  image.cols, image.rows);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);

    cout<<"time is"<<elapsedTime<<std::endl;
    cudaMemcpy(out.data, dev_out, image.rows * image.cols* 3, cudaMemcpyDeviceToHost);
    imwrite("blur.jpg", out);
    cudaFree(dev_bitmap);
    cudaFree(dev_out);
    return 0;
}