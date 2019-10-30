#include <iostream>
#include <cstdlib>
#include <cfloat>
#include <math.h>
#include <sys/time.h>

#define THREADS_PER_BLOCK 32

__global__ void voronoi_d (int *imageArray, int *points, int imageSize, int numPoints) {
    // use x to access each cell and compare it to each point and assign the cell's value to match the closest point
    int x = blockIdx.x * blockDim.x + threadIdx.x;

    double minDistance = DBL_MAX;
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
    // allocate space for the image and point coordinates on the device and copy the coordinates over
    int *imageArray;
    int *points;
	cudaMalloc ((void**) &imageArray, sizeof(int) * imageSize * imageSize);
	cudaMalloc ((void**) &points, sizeof(int) * numPoints * 2);
    cudaMemcpy (points, points_h, sizeof(int) * numPoints * 2, cudaMemcpyHostToDevice);

    // start calculation timing
    struct timeval start, end;
    gettimeofday(&start, NULL);

    // calculate and then synchronize to ensure accurate timing
    voronoi_d <<< ceil((float) imageSize*imageSize/THREADS_PER_BLOCK), THREADS_PER_BLOCK >>> (imageArray, points, imageSize, numPoints);
    cudaDeviceSynchronize();

    // end timing and print processing time
    gettimeofday(&end, NULL);
    long seconds = (end.tv_sec - start.tv_sec);
    long micros = ((seconds * 1000000) + end.tv_usec) - (start.tv_usec);
    printf("Processing time elpased is %zu seconds or %zu micros\n", seconds, micros);

    // print CUDA errors
    cudaError_t err = cudaGetLastError();
    printf("CUDA error: %s\n", cudaGetErrorString(err));

    // copy results to host and free device memory
    cudaMemcpy (imageArray_h, imageArray, sizeof(int) * imageSize * imageSize, cudaMemcpyDeviceToHost);
    cudaFree (imageArray);
    cudaFree (points);
}

