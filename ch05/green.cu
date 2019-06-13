//
// Created by zhangjian on 19-6-13.
//
#include "../common/book.h"
#include "../common/image.h"


const int DIM = 1024;
const float PI(3.1415926535897932f);
__global__ void kernel(unsigned char *ptr){


    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    int offset = x + y * blockDim.x * gridDim.x;

    __shared__ float shared[16][16];

    const float period = 128.0f;
    shared[threadIdx.x][threadIdx.y] =
            255 * (sinf(x * 2.0f * PI / period) + 1.0f) *
                    (sinf(y * 2.0f * PI / period) + 1.0f) / 4.0f;
    __syncthreads();

    ptr[offset*4 + 0] = 0;
    ptr[offset*4 + 1] = shared[15 - threadIdx.x][15 - threadIdx.y];
    ptr[offset*4 + 2] = 0;
    ptr[offset*4 + 3] = 255;
}


int main(void){

    IMAGE bitmap(DIM, DIM);
    unsigned char *dev_bitmap;
    dim3 grid(DIM / 16, DIM / 16);
    dim3 threads(16, 16);

    cudaMalloc( (void**)&dev_bitmap, bitmap.image_size());

    kernel<<<grid, threads>>>(dev_bitmap);

    cudaMemcpy(bitmap.get_ptr(), dev_bitmap,
             bitmap.image_size(),
             cudaMemcpyDeviceToHost);
    bitmap.show_image();
    cudaFree(dev_bitmap);

}
