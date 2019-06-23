//
// Created by zhangjian on 19-6-23.
//

#include "../common/book.h"
#include <iostream>

#define imin(a,b) (a<b?a:b)

const int N = 33 * 1024;
const int threadsPerBlock = 256;
const int blocksPerGrid =
        imin( 32, (N+threadsPerBlock-1) / threadsPerBlock );

__global__ void dot( float *a, float *b, float *c ) {
    __shared__ float cache[threadsPerBlock];
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int cacheIndex = threadIdx.x;

    float   temp = 0;
    while (tid < N) {
        temp += a[tid] * b[tid];
        tid += blockDim.x * gridDim.x;
    }

    // set the cache values
    cache[cacheIndex] = temp;

    // synchronize threads in this block
    __syncthreads();

    // for reductions, threadsPerBlock must be a power of 2
    // because of the following code
    int i = blockDim.x/2;
    while (i != 0) {
        if (cacheIndex < i)
            cache[cacheIndex] += cache[cacheIndex + i];
        __syncthreads();
        i /= 2;
    }

    if (cacheIndex == 0)
        c[blockIdx.x] = cache[0];
}

float malloc_test(int size){

    cudaEvent_t start, stop;
    float *a, *b, c, *partial_c;
    float *dev_a, *dev_b, *dev_partial_c;
    float elapsedTime;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    a = (float*)malloc(size * sizeof(float));
    b = (float*)malloc(size * sizeof(float));
    partial_c = (float*)malloc(blocksPerGrid * sizeof(float));

    cudaMalloc( ( void**)&dev_a, size * sizeof(float));
    cudaMalloc( ( void**)&dev_b, size * sizeof(float));
    cudaMalloc( ( void**)&dev_partial_c, blocksPerGrid * sizeof(float));

    for(int i = 0; i < size; i++){
        a[i] = i;
        b[i] = i * 2;
    }

    cudaEventRecord(start, 0);

    cudaMemcpy(dev_a, a, size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, b, size * sizeof(float), cudaMemcpyHostToDevice);

    dot<<<blocksPerGrid, threadsPerBlock>>>(dev_a, dev_b, dev_partial_c);

    cudaMemcpy( partial_c, dev_partial_c,
                blocksPerGrid*sizeof(float),
                cudaMemcpyDeviceToHost );
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);

    c =0;
    for(int i = 0; i < blocksPerGrid;i++){
        c += partial_c[i];
    }

    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_partial_c);
    free(a);
    free(b);
    free(partial_c);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return elapsedTime;
}


float cuda_host_alloc_test(int size){

    cudaEvent_t start, stop;
    float *a, *b, c, *partial_c;
    float *dev_a, *dev_b, *dev_partial_c;
    float elapsedTime;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaHostAlloc((void**)&a, size* sizeof(float), cudaHostAllocWriteCombined | cudaHostAllocMapped);
    cudaHostAlloc((void**)&b, size* sizeof(float), cudaHostAllocWriteCombined | cudaHostAllocMapped);
    cudaHostAlloc((void**)&partial_c, blocksPerGrid* sizeof(float), cudaHostAllocMapped);



    for(int i = 0; i < size; i++){
        a[i] = i;
        b[i] = i * 2;
    }

    cudaHostGetDevicePointer(&dev_a, a, 0);
    cudaHostGetDevicePointer(&dev_b, b, 0);
    cudaHostGetDevicePointer(&dev_partial_c, partial_c, 0);

    cudaEventRecord(start, 0);

    dot<<<blocksPerGrid, threadsPerBlock>>>(dev_a, dev_b, dev_partial_c);

    cudaMemcpy( partial_c, dev_partial_c,
                blocksPerGrid*sizeof(float),
                cudaMemcpyDeviceToHost );
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);

    c =0;
    for(int i = 0; i < blocksPerGrid;i++){
        c += partial_c[i];
    }

    cudaFreeHost(a);
    cudaFreeHost(b);
    cudaFreeHost(partial_c);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return elapsedTime;
}


int main(void){

    cudaDeviceProp prop;
    int whichDevice;
    int deviceCount = 0;
    cudaGetDevice(&whichDevice);
    cudaGetDeviceProperties(&prop, whichDevice);
    cudaGetDeviceCount(&deviceCount);
    std::cout<<deviceCount<<std::endl;
    if(prop.canMapHostMemory != 1){
        std::cout<<"Device cannot map memory\n"<<std::endl;
        return 0;
    }

    cudaSetDeviceFlags(cudaDeviceMapHost);

    float elapsedTime = malloc_test(N);
    std::cout<<"Time using cudamalloc: "<< elapsedTime<<std::endl;

    elapsedTime = cuda_host_alloc_test(N);
    std::cout<<"Time using cduaHostAlloc: "<<elapsedTime<<std::endl;
}

































