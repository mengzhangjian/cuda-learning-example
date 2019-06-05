#include "../common/book.h"
#include <iostream>

const int N = 10;

__global__ void add(int *a, int *b, int *c){

    int tid = threadIdx.x;

    if(tid < N)
        c[tid] = a[tid] + b[tid];
}

int main(void){

    int a[N], b[N], c[N];

    int *dev_a, *dev_b, *dev_c;

    cudaMalloc( (void**)&dev_a, sizeof(int) * N );
    cudaMalloc( (void**)&dev_b, sizeof(int) * N);
    cudaMalloc( (void**)&dev_c, sizeof(int) * N );

    for(int i = 0; i < N; i++){
        a[i] = -i;
        b[i] = i;
    }

    cudaMemcpy(dev_a, a, N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, b, N * sizeof(int), cudaMemcpyHostToDevice);

    add<<<1, N>>>(dev_a, dev_b, dev_c);

    cudaMemcpy(c, dev_c, sizeof(int) * N, cudaMemcpyDeviceToHost);
    for(int i = 0; i < N; i++)
        std::cout << a[i] << std::endl;
    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);
}