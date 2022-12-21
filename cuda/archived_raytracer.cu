//#include <png.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <sstream>
//#include <zlib.h>
#include <string>
#include <png++/png.hpp>
#include "vec2.h"
#include <math.h>
#include "timing.h"
#include "common.h"
#include <cmath>

#define NUMTHREADS 1024

__device__ bool rayInBox(float tlx, float tly, float brx, float bry, float posX, float posY, int rayIdx) {
  float brHeadX = brx - posX;
  float brHeadY = bry - posY;
  float tlHeadX = posX - tlx;
  float tlHeadY = posY - tly;
  bool result = (tlHeadX >= 0 && tlHeadY >= 0 && brHeadX >= 0 && brHeadY >= 0);
  return result;
}

__device__ bool almostEqual(float a, float b) {
  return (abs(b - a) <= 10e-12);
}

float computeContributionFactor(float score) {
  if (score < L1_THRESHOLD) return 0.f;
  else if (score < L2_THRESHOLD) return tuningCnst + tuningScale * log(score);
  else return 1.f;
}

float sumFloats(std::vector<float*> &multiScores, int r, int c, int cols, int n) {
  float ans = 0.f;
  for (int i = 0; i < n; i++) {
    float *accessPoint = multiScores[i] + (r * cols) + c;
    ans += *accessPoint;
  }
  return ans;
}

void colorOutput(png::image<png::rgb_pixel> &colorImg, std::vector<float*> &multiScores, std::vector<lightray> lightSources, int rows, int cols, int numSources) {
    // apply source light to image

  for (int r = 0; r < rows; r++) {
    for (int c = 0; c < cols; c++) {
      float pixelScore = sumFloats(multiScores, r, c, cols, numSources);

      float effRed = 0;
      float effGreen = 0;
      float effBlue = 0;
      for (int s = 0; s < numSources; s++) {
        float *accessPoint = multiScores[s] + (r * cols) + c;
        float sourceCont = *accessPoint;
        effRed += lightSources[s].color.red * sourceCont / pixelScore;
        effBlue += lightSources[s].color.blue * sourceCont / pixelScore;
        effGreen += lightSources[s].color.green * sourceCont / pixelScore;
      }

      png::rgb_pixel oneLight = png::rgb_pixel((int)effRed, (int)effGreen, (int)effBlue);

      int contributionRed = (oneLight.red - colorImg[r][c].red) * computeContributionFactor(pixelScore); 
      int contributionGreen = (oneLight.green - colorImg[r][c].green) * computeContributionFactor(pixelScore);
      int contributionBlue = (oneLight.blue - colorImg[r][c].blue) * computeContributionFactor(pixelScore);

      int newRed = std::min(255, colorImg[r][c].red + contributionRed);
      int newBlue = std::min(255, colorImg[r][c].blue + contributionBlue);
      int newGreen = std::min(255, colorImg[r][c].green + contributionGreen);
      
      //apply
      colorImg[r][c] = png::rgb_pixel(newRed, newGreen, newBlue);
    }
  }
}

__device__ void rotate(float* x, float* y, double deg) {
  double theta = deg / 180.0 * M_PI;
  double c = cos(theta);
  double s = sin(theta);
  double tx = (*x) * c - (*y) * s;
  double ty = (*x) * s + (*y) * c;
  *x = tx;
  *y = ty;
}

__device__ char readTrace(char *tracedImg, int row, int col, int cols) {
  return *(tracedImg + (row * cols) + col);
}

__device__ float distance(float x1, float y1, float x2, float y2) {
  float delX = x1 - x2;
  float delY = y1 - y2;
  return sqrt((delX * delX) + (delY * delY));
}

// fills in the scores according to one light source
__global__ void fillPartialScores(float *cudaScores, lightray source, int rows, int cols, char *tracedImg) {

  int rayIdx = blockIdx.x * blockDim.x + threadIdx.x;

  if (rayIdx >= RAYCOUNT) {
    return; // remove extra threads
  }
  // ray initialization
  float tlx = 0.0f;
  float tly = 0.0f;
  float brx = (float)cols;
  float bry = (float)rows;
  float curDirX = 1.0f;
  float curDirY = 0.0f;
  
  double rotAmt = (double)rayIdx * (360.0 / RAYCOUNT);
  rotate(&curDirX, &curDirY, rotAmt);
  lightray mRay;
  mRay.position.x = source.position.x;
  mRay.position.y = source.position.y;
  mRay.velocity.x = curDirX;
  mRay.velocity.y = curDirY;
  mRay.intensity = source.intensity;

  int propogations = 0; // keep score for debugging
  while (rayInBox(tlx, tly, brx, bry, mRay.position.x, mRay.position.y, rayIdx)) {
    if (readTrace(tracedImg, (int)mRay.position.y, (int)mRay.position.x, cols)  == 0) break;
    else { // update ray
      //updateRay(&lightSource, &mRay, singleScores, cols, numSources, sourceNum); // update the ray based on a given source
      if (!(almostEqual(mRay.position.x, source.position.x) && almostEqual(mRay.position.y, source.position.y))) {
        int accessRow = (int)mRay.position.y;
        int accessCol = (int)mRay.position.x;

        float *accessPoint = cudaScores + (accessRow * cols) + accessCol;
        float incrementVal = mRay.intensity;
        atomicAdd(accessPoint, incrementVal); // threads acknowledge each other's contributions
        //if (rayIdx == 1350) printf("largeScore: %f, rowcol: (%d, %d), propogations: %d, incrementVal: %.2f\n", *(accessPoint), accessRow, accessCol, propogations, incrementVal);
      }
      // update position and intensity
      float deltaX = mRay.velocity.x * incrRay;
      float deltaY = mRay.velocity.y * incrRay;

      mRay.position.x += deltaX;
      mRay.position.y += deltaY;
      float normDist = distance(mRay.position.x, mRay.position.y, source.position.x, source.position.y) / incrRay; // normalized distance
      mRay.intensity = addFac * source.intensity / (normDist * normDist);

      propogations += 1;
    }
  }
  //if (rayIdx == 1040) printf("largeScore: %f... %d\n", *(cudaScores), propogations);
  return;
}

// takes in <color.png> <traced.png> and produces an <output.png>
int main(int argc, char** argv) {
  if (argc != 5) {
    printf("Usage: raytrace <(color)*.png> <(traced)*.png> <(lights)*.txt> <(output)*.png>\n");
    return 1;
  } 

  //const float defaultDistPixel = 1.0f;

  std::vector<lightray> lightSources;
  loadFromFile(argv[3], lightSources);
    
  png::image<png::rgb_pixel> colorImg(argv[1]);
  png::image<png::gray_pixel> tracedImg(argv[2]);
  int rows = colorImg.get_height();
  int cols = colorImg.get_width();

  if (rows != tracedImg.get_height() && cols != tracedImg.get_width()) {
    printf("Error: Color Image and Traced Image are not of same dimension\n");
    return 2;
  }

  int numSources = lightSources.size();
  std::vector<float*> cudaScoreElements;
  std::vector<float*> multiScores;
  multiScores.reserve(numSources);
  int pixelCount = rows * cols;
  size_t frameSize = pixelCount * sizeof(float);

  int traceBufSize = sizeof(char) * pixelCount;
  char* tracedImageRaw = (char*)malloc(traceBufSize);
  // copy into raw bytes
  for (int r = 0; r < rows; r++) {
    for (int c = 0; c < cols; c++) {
      *(tracedImageRaw + (r * cols) + c) = tracedImg[r][c];
    }
  }
  char* cudaTrace;

 Timer rayTracerTimer;

  cudaMalloc(&cudaTrace, traceBufSize);
  cudaMemcpy(cudaTrace, tracedImageRaw, traceBufSize, cudaMemcpyHostToDevice);
  cudaMemset(cudaTrace, 0, traceBufSize);
  cudaMemcpy(cudaTrace, tracedImageRaw, traceBufSize, cudaMemcpyHostToDevice);
   
  for (int s = 0; s < numSources; s++) {
    float* sourceScores = (float*)calloc(rows * cols, sizeof(float));
    multiScores[s] = sourceScores;
    lightray source = lightSources[s];
    float* cudaScores;
    cudaMalloc(&cudaScores, frameSize);
    cudaMemset(cudaScores, 0, frameSize);
    cudaScoreElements.push_back(cudaScores);
    // kernel time
    int blocks = (RAYCOUNT + NUMTHREADS - 1) / NUMTHREADS;
    fillPartialScores<<<blocks,NUMTHREADS>>>(cudaScores, source, rows, cols, cudaTrace);
  }

  cudaDeviceSynchronize();

  for (int s = 0; s < numSources; s++) {
    cudaMemcpy(multiScores[s], cudaScoreElements[s], frameSize, cudaMemcpyDeviceToHost); 
    cudaFree(cudaScoreElements[s]); 
  }

  cudaFree(cudaTrace);
  free(tracedImageRaw);
  double rayTracerTime1 = rayTracerTimer.elapsed();

  //void* singleScores = calloc(rows * cols * numSources, sizeof(float));
  //float singleScores[rows][cols][numSources] = { 0 };

  // accumulate the contributions from each partial score onto the main image
  colorOutput(colorImg, multiScores, lightSources, rows, cols, numSources);

  // free partial scores
  for (int s = 0; s < numSources; s++) {
    free(multiScores[s]);
  }

  double rayTracerTime2 = rayTracerTimer.elapsed();
  printf("total cuda raytracer time: %.6fs\n", rayTracerTime2);

  colorImg.write(argv[4]); // finished product
  //free(singleScores);
  return 0;
}