#include <iostream>
#include <fstream>
#include <cmath>
#include <chrono>
#include <cstdlib>
#include <map>
#include <vector>
#include <random>
#include <algorithm>
#include <cfloat>
#include <math.h>


#define THREADS_PER_BLOCK 256

__global__ void voronoi_d (int *imageArray, int *points, int imageSize, int numPoints) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;

    double minDistance = 1000000;
    int minPoint = -1;
    for (int k=0; k<numPoints; k++) {
        double distance = sqrt(pow((double) (x % imageSize - points[k + numPoints]), 2.0) + pow((double) (x / imageSize - points[k]), 2.0));
        if (distance < minDistance) {
            minDistance = distance;
            minPoint = k;
        }
    }
    imageArray[x] = minPoint;
}

extern void gpuVoronoi(int *imageArray_h, int *points_h, int imageSize, int numPoints)
{
    printf("starting host code \n");

    int *imageArray;
    int *points;

	cudaMalloc ((void**) &imageArray, sizeof(int) * imageSize * imageSize);
	cudaMalloc ((void**) &points, sizeof(int) * numPoints * 2);
    cudaMemcpy (points, points_h, sizeof(int) * numPoints * 2, cudaMemcpyHostToDevice);

    voronoi_d <<< ceil((float) imageSize*imageSize/THREADS_PER_BLOCK), THREADS_PER_BLOCK >>> (imageArray, points, imageSize, numPoints);

    cudaError_t err = cudaGetLastError();
    printf("CUDA error: %s\n", cudaGetErrorString(err));

    cudaMemcpy (imageArray_h, imageArray, sizeof(int) * imageSize * imageSize, cudaMemcpyDeviceToHost);
    cudaFree (imageArray);
    cudaFree (points);

    printf("ending host code \n");
}

