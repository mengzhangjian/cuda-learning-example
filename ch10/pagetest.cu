//
// Created by zhangjian on 19-6-22.
//
#include "../common/book.h"
#include "cuda.h"
#include <iostream>
#define  SIZE (10*1024*1024)
float cuda_malloc_test(int size, bool up){


    int *a, *dev_a;
    float elapsedTime;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    a = (int*)malloc(size * sizeof(*a));
    cudaMalloc((void**)&dev_a, size * sizeof(*dev_a));

    cudaEventRecord(start, 0);
    for(int i = 0; i < 100; i++){

        if(up)
            cudaMemcpy(dev_a, a, size * sizeof(*a), cudaMemcpyHostToDevice);
        else
            cudaMemcpy(a, dev_a, size * sizeof(*dev_a), cudaMemcpyDeviceToHost);
    }
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    cudaEventElapsedTime(&elapsedTime, start, stop);
    free(a);
    cudaFree(dev_a);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return elapsedTime;
}

float cuda_malloc_host_test(int size, bool up){

    int *a, *dev_a;
    float elapsedTime;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaHostAlloc( (void**)&a, size * sizeof(*a), cudaHostAllocDefault);
    cudaMalloc((void**)&dev_a, size * sizeof(*dev_a));

    cudaEventRecord(start, 0);
    for(int i = 0; i < 100; i++){

        if(up)
            cudaMemcpy(dev_a, a, size * sizeof(*a), cudaMemcpyHostToDevice);
        else
            cudaMemcpy(a, dev_a, size * sizeof(*dev_a), cudaMemcpyDeviceToHost);
    }
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    cudaEventElapsedTime(&elapsedTime, start, stop);
    cudaFreeHost(a);
    cudaFree(dev_a);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return elapsedTime;
}

int main(void){

    float elapsedTime;
    float MB = (float)100*SIZE* sizeof(int)/1024/1024;
    elapsedTime = cuda_malloc_test(SIZE, true);

    std::cout<<elapsedTime<<std::endl;
    elapsedTime = cuda_malloc_host_test(SIZE, true);
    std::cout<<elapsedTime<<std::endl;
}