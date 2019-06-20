//
// Created by Zhang,Jian(XBU-UPD) on 2019-06-20.
//
#include "../common/book.h"
#include "cuda.h"
#include <iostream>

#define  SIZE (100 * 1024 * 1024)

int main(void){
    unsigned  char *buffer = (unsigned char *) big_random_block(SIZE);

    unsigned int histo[256];
    for(int i = 0; i < 256; i++){
        histo[i] = 0;
    }
    for(int i = 0; i < SIZE; i++)
        histo[buffer[i]]++;

    long histoCount = 0;
    for(int i = 0; i < 256; i++){
        histoCount += histo[i];
    }

    std::cout<<"Sum is " << histoCount << std::endl;
    free(buffer);
    return 0;
}