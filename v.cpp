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

extern void gpuVoronoi(int *imageArray, int *points, int imageSize, int numPoints);
 
using namespace std;
int main (int argc, char *argv[]) {

    // check to make sure we have correct number of arguments
    if (argc != 3) {
        cerr<< "invalid number of arguments" << "\n";
        exit (-1);
    }

    // get the image size and number of points from the arguments
    int imageSize = atoi(argv[1]);
    int numPoints = atoi(argv[2]);

    // create array to hold the calculation results
    int* imageArray = new int[imageSize * imageSize];

    // create vectors of possible X and Y coordinates for the points
    vector<int> pointsX;
    vector<int> pointsY;

    // load the vectors with all possible X and Y values
    for(int i=0; i<imageSize; i++) {
        pointsX.push_back(i);
        pointsY.push_back(i);
    }

    // shuffle the coordinate vectors
    unsigned seedX = std::chrono::system_clock::now().time_since_epoch().count();
    shuffle(pointsX.begin(), pointsX.end(), default_random_engine(seedX));
    unsigned seedY = std::chrono::system_clock::now().time_since_epoch().count();
    shuffle(pointsY.begin(), pointsY.end(), default_random_engine(seedY));

    // fill the points with random coordinates
    int* points = new int[numPoints + numPoints];
    for(int i = 0; i < numPoints; i++) {
        points[i] = pointsX[i];
        points[i + numPoints] = pointsY[i];
    }


    // create maps to hold the RGB values of each point
    map<int, int> pointRed;
    map<int, int> pointGreen;
    map<int, int> pointBlue;

    // set the final entry to white to color the points against the cells
    pointRed[numPoints] = 255;
    pointGreen[numPoints] = 255;
    pointBlue[numPoints] = 255;

    // populate maps with random colors
    for (int i=0; i<numPoints; i++) {
        pointRed[i] = rand() % 256;
        pointGreen[i] = rand() % 256;
        pointBlue[i] = rand() % 256;
    }

    // populate the imageArray on the GPU
    gpuVoronoi(imageArray, points, imageSize, numPoints);
    

    // loop through points and set color to white (numPoints)
    for (int i=0; i<numPoints; i++) {
        imageArray[points[i + numPoints] + imageSize * points[i]] = numPoints;
    }

    // write points to file by pulling colors from maps
    ofstream output("C:/Users/jeffp/CLionProjects/HPC3/output.ppm", ios_base::binary);

    // store the image header in our output file
    output << "P3" << "\n";
    output << imageSize << " " << imageSize << "\r\n";
    output << "255" << "\r\n";

    // store the updated array in our output file
    for(int i = 0; i < imageSize * imageSize; i++) {
        output << pointRed[imageArray[i]] << " " << pointGreen[imageArray[i]] << " " << pointBlue[imageArray[i]] << " ";
        output << "\r\n";
    }
    output.close();

    // delete arrays from memory
    delete[] imageArray;
    delete[] points;

    return 0;
}