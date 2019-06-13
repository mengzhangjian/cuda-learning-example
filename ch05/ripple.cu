#include "../common/book.h"
#include "../common/image.h"

const int DIM(1024);
const float PI(3.1415926535897932);

__global__ void kernel(unsigned char *ptr, int ticks){

    //map from threadIdx / BlockIdx to pixel position

    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int offset = x + y * blockDim.x * gridDim.x;

    //calculate the value at that position
    float fx = x - DIM/2;
    float fy = y - DIM/2;
    float d = sqrtf( fx * fx + fy * fy );
    unsigned char grey = (unsigned char)(128.0f + 127.0f *
                                                  cos(d/10.0f - ticks/7.0f) /
                                                  (d/10.0f + 1.0f));
    ptr[offset*4 + 0] = grey;
    ptr[offset*4 + 1] = grey;
    ptr[offset*4 + 2] = grey;
    ptr[offset*4 + 3] = 255;
}

struct  DataBlock{
    unsigned  char *dev_bitmap;
    IMAGE *bitmap;
};

void cleanUp( DataBlock *d){
    cudaFree(d->dev_bitmap);
}

void generate_frame(DataBlock *d, int ticks){

    dim3 blocks(DIM / 16, DIM / 16);
    dim3 threads(16, 16);

    kernel<<<blocks, threads>>>(d->dev_bitmap, ticks);

    cudaMemcpy(d->bitmap->get_ptr(), d->dev_bitmap, d->bitmap->image_size(),
            cudaMemcpyDeviceToHost);

}

int main(void){
    DataBlock data;
    IMAGE bitmap(DIM, DIM);
    data.bitmap = &bitmap;

    cudaMalloc((void**) &data.dev_bitmap, bitmap.image_size());

    dim3 blocks(DIM / 16, DIM / 16);
    dim3 threads(16, 16);

    int tricks = 0;
    bitmap.show_image(30);
    while(1){
        generate_frame(&data, tricks);
        tricks++;
        char key = bitmap.show_image(30);
        if(key == 27)
        {
            break;
        }

    }
    cleanUp(&data);


}
