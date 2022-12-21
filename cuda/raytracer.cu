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
#define COLOR_COMPONENTS 3
#define REDMOD 0
#define GREENMOD 1
#define BLUEMOD 2

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

__device__ float computeContributionFactor(float score) {
  if (score < L1_THRESHOLD) return 0.f;
  else if (score < L2_THRESHOLD) return tuningCnst + tuningScale * log(score);
  else return 1.f;
}

__device__ float sumFloats(float *fs, int n) {
  float ans = 0.f;
  for (int i = 0; i < n; i++) {
    ans += *(fs + i);
  }
  return ans;
}

void colorOutput(unsigned char* hostColors, int rows, int cols, png::image<png::rgb_pixel> &colorImg) {
    // apply source light to image

  for (int r = 0; r < rows; r++) {
    for (int c = 0; c < cols; c++) {
      int newRed = *(hostColors + (r * cols * COLOR_COMPONENTS) + (c * COLOR_COMPONENTS) + REDMOD);
      int newBlue = *(hostColors + (r * cols * COLOR_COMPONENTS) + (c * COLOR_COMPONENTS) + BLUEMOD);
      int newGreen = *(hostColors + (r * cols * COLOR_COMPONENTS) + (c * COLOR_COMPONENTS) + GREENMOD);
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

// stores effective rgb values into cudaColors based on partialCudaScores data
__global__ void produceColors(unsigned char *cudaColors, float **partialCudaScores, unsigned char* loadedCudaColors, int numSources, 
                              unsigned char* cudaColorRaw, int rows, int cols) {
  int pixelIndex = blockIdx.x * blockDim.x + threadIdx.x; // spot in linear pixel memory
  int redIndex = COLOR_COMPONENTS * pixelIndex + REDMOD;
  int greenIndex = COLOR_COMPONENTS * pixelIndex + GREENMOD;
  int blueIndex = COLOR_COMPONENTS * pixelIndex + BLUEMOD;

  if (pixelIndex >= rows*cols) {
    return; // remove extra threads
  }

  float pixelScore = 0;
  for (int s = 0; s < numSources; s++) { // derive total score from all partials
    float *partialScoresArr = *(partialCudaScores + s);
    pixelScore += *(partialScoresArr + pixelIndex);
  } 

  float effRed = 0;
  float effGreen = 0;
  float effBlue = 0;
  for (int s = 0; s < numSources; s++) {
    unsigned char redSource = *(loadedCudaColors + (s * COLOR_COMPONENTS) + REDMOD);
    unsigned char greenSource = *(loadedCudaColors + (s * COLOR_COMPONENTS) + GREENMOD);
    unsigned char blueSource = *(loadedCudaColors + (s * COLOR_COMPONENTS) + BLUEMOD);
    float *partialScoresArr = *(partialCudaScores + s);
    float scoresCont = *(partialScoresArr + pixelIndex);

    effRed += redSource * scoresCont / pixelScore;
    effGreen += greenSource * scoresCont / pixelScore;
    effBlue += blueSource * scoresCont / pixelScore;
  }

  int contributionRed = (((int)effRed) - (*(cudaColorRaw + redIndex))) * computeContributionFactor(pixelScore);
  int contributionGreen = (((int)effGreen) - (*(cudaColorRaw + greenIndex))) * computeContributionFactor(pixelScore);
  int contributionBlue = (((int)effBlue) - (*(cudaColorRaw + blueIndex))) * computeContributionFactor(pixelScore);

  *(cudaColors + redIndex) = min(255, (*(cudaColorRaw + redIndex)) + contributionRed);
  *(cudaColors + greenIndex) = min(255, (*(cudaColorRaw + greenIndex)) + contributionGreen);
  *(cudaColors + blueIndex) = min(255, (*(cudaColorRaw + blueIndex)) + contributionBlue);

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

  int scoresHoldSize = sizeof(float*) * numSources;
  float** multiCudaScores = (float**)malloc(scoresHoldSize);
   
  for (int s = 0; s < numSources; s++) {
    lightray source = lightSources[s];
    float* cudaScores;
    cudaMalloc(&cudaScores, frameSize);
    cudaMemset(cudaScores, 0, frameSize);
    multiCudaScores[s] = cudaScores;
    // kernel time
    int blocks = (RAYCOUNT + NUMTHREADS - 1) / NUMTHREADS;
    fillPartialScores<<<blocks,NUMTHREADS>>>(cudaScores, source, rows, cols, cudaTrace);
  }

  cudaDeviceSynchronize(); // synchronize partial scores
  // cudaMemcpy(sourceScores, cudaScores, frameSize, cudaMemcpyDeviceToHost); // copy result outside of loop
  //  cudaFree(cudaScores); // don't free just yet actually

  cudaFree(cudaTrace);
  free(tracedImageRaw);

  float** partialCudaScores;
  cudaMalloc(&partialCudaScores, scoresHoldSize);
  cudaMemcpy(partialCudaScores, multiCudaScores, scoresHoldSize, cudaMemcpyHostToDevice);

  // allocate sources memory for cuda
  int colorsMemSize = sizeof(char) * COLOR_COMPONENTS * numSources;
  unsigned char* loadedColors = (unsigned char*)malloc(colorsMemSize);
  for (int s = 0; s < numSources; s++) {
    int colorIdx = s * COLOR_COMPONENTS;
    loadedColors[colorIdx + REDMOD] = lightSources[s].color.red;
    loadedColors[colorIdx + GREENMOD] = lightSources[s].color.green;
    loadedColors[colorIdx + BLUEMOD] = lightSources[s].color.blue;
  }

  unsigned char* loadedCudaColors;
  cudaMalloc(&loadedCudaColors, colorsMemSize);
  cudaMemcpy(loadedCudaColors, loadedColors, colorsMemSize, cudaMemcpyHostToDevice);
  free(loadedColors);

  // compute actual color components for each contribution pixel
  unsigned char* cudaColors; // layed out in memory as r0_g0_b0, r1_g1_b1, ...
  unsigned char* hostColors;
  cudaMalloc(&cudaColors, traceBufSize * COLOR_COMPONENTS);
  hostColors = (unsigned char*)malloc(traceBufSize * COLOR_COMPONENTS);

  unsigned char* colorImageRaw = (unsigned char*)malloc(traceBufSize * COLOR_COMPONENTS);
  // copy into raw bytes
  for (int r = 0; r < rows; r++) {
    for (int c = 0; c < cols; c++) {
      *(colorImageRaw + (r * cols * COLOR_COMPONENTS) + (c * COLOR_COMPONENTS) + REDMOD) = colorImg[r][c].red;
      *(colorImageRaw + (r * cols * COLOR_COMPONENTS) + (c * COLOR_COMPONENTS) + GREENMOD) = colorImg[r][c].green;
      *(colorImageRaw + (r * cols * COLOR_COMPONENTS) + (c * COLOR_COMPONENTS) + BLUEMOD) = colorImg[r][c].blue;
    }
  }
  unsigned char* cudaColorRaw;
  cudaMalloc(&cudaColorRaw, traceBufSize * COLOR_COMPONENTS);
  cudaMemcpy(cudaColorRaw, colorImageRaw, traceBufSize * COLOR_COMPONENTS, cudaMemcpyHostToDevice);
  free(colorImageRaw);

  int colorBlocks = (pixelCount + NUMTHREADS - 1) / NUMTHREADS;
  produceColors<<<colorBlocks,NUMTHREADS>>>(cudaColors, partialCudaScores, loadedCudaColors, numSources, cudaColorRaw, rows, cols);
  cudaDeviceSynchronize();

  cudaFree(cudaColorRaw);
  cudaFree(loadedCudaColors);
  cudaMemcpy(hostColors, cudaColors, traceBufSize * COLOR_COMPONENTS, cudaMemcpyDeviceToHost);
  cudaFree(cudaColors);
  cudaFree(partialCudaScores);
  for (int s = 0; s < numSources; s++) {
    float *cudaScore = multiCudaScores[s];
    cudaFree(cudaScore);
  }
  free(multiCudaScores);

  // accumulate the contributions from each partial score onto the main image
  colorOutput(hostColors, rows, cols, colorImg);
  free(hostColors);

  double rayTracerTime = rayTracerTimer.elapsed();
  printf("total raytracer time: %.6fs\n", rayTracerTime);

  colorImg.write(argv[4]); // finished product

  return 0;
}