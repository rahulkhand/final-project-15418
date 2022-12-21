#include <png.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <sstream>
#include <vector>
#include <zlib.h>
#include <string>
#include <png++/png.hpp>
#include "vec2.h"
#include "/usr/local/opt/libomp/include/omp.h"
#include <math.h>
#include "timing.h"
#include "common.h"

bool rayInBox(Vec2<float> tl, Vec2<float> br, lightray mRay) {
  Vec2<float> brHead = br - (mRay.position);
  Vec2<float> tlHead = (mRay.position) - tl;
  return (tlHead.x >= 0 && tlHead.y >= 0 && brHead.x >= 0 && brHead.y >= 0);
}

bool almostEqual(float a, float b) {
  return (abs(b - a) <= 10e-12);
}

// updates the position and intensity of a ray while contributing its partial score to the new image
void updateRay(lightray *source, lightray *mRay, std::vector<float> &ss, int cols, int numSources, int sourceNum) {
  if (!(almostEqual(mRay->position.x, source->position.x) && almostEqual(mRay->position.y, source->position.y))) {
    int accessRow = (int)mRay->position.y;
    int accessCol = (int)mRay->position.x;

    int accessIdx = (accessRow * cols * numSources) + (accessCol * numSources) + sourceNum;
    ss[accessIdx] += mRay->intensity;
  }
  // update position and intensity
  Vec2f delta = mRay->velocity * incrRay;
  mRay->position.x += delta.x;
  mRay->position.y += delta.y;
  float normDist = mRay->position.dist(source->position) / incrRay; // normalized distance
  mRay->intensity = addFac * source->intensity / (normDist * normDist);
  return;
}

float computeContributionFactor(float score) {
  const float tuningCnst = 0.0087209302; // constants derived by fitting to logarithmic function
  const float tuningScale = 0.0820614573;
  if (score < L1_THRESHOLD) return 0.f;
  else if (score < L2_THRESHOLD) return tuningCnst + tuningScale * log(score);
  else return 1.f;
}

// casts rays from a single source identified by sourceNum to update singleScores
void fillPartialScores(std::vector<float> &singleScores, int sourceNum, lightray lightSource, int rows, int cols, int numSources,
                    png::image<png::gray_pixel> tracedImg) {
  const Vec2f tl(0.0f, 0.0f);
  const Vec2f br((float)cols, (float)rows);
  const Vec2f unitVecX(1.0f, 0.0f);

  // spawn rays from source
  for (int i = 0; i < RAYCOUNT; i++) {
    Vec2f curDir(unitVecX);
    double rotAmt = (double)i * (360.0 / RAYCOUNT);
    curDir.rotate(rotAmt);
    lightray mRay;
    mRay.position = lightSource.position;
    mRay.velocity = curDir;
    mRay.intensity = lightSource.intensity;

    while (rayInBox(tl, br, mRay)) {
      if (tracedImg[(int)mRay.position.y][(int)mRay.position.x] == 0) break;
      else {
        updateRay(&lightSource, &mRay, singleScores, cols, numSources, sourceNum); // update the ray based on a given source
      }
    }
  }
}

float sumFloats(std::vector<float> &fs, int accessIdx, int n) {
  float ans = 0.f;
  for (int i = 0; i < n; i++) {
    ans += fs[accessIdx + i];
  }
  return ans;
}

void colorOutput(png::image<png::rgb_pixel> &colorImg, std::vector<float> &ss, std::vector<lightray> lightSources, int rows, int cols, int numSources) {
    // apply source light to image

  for (int r = 0; r < rows; r++) {
    for (int c = 0; c < cols; c++) {
      int accessIdx = (r * cols * numSources) + (c * numSources);
      float pixelScore = sumFloats(ss, accessIdx, numSources);

      float effRed = 0;
      float effGreen = 0;
      float effBlue = 0;
      for (int s = 0; s < numSources; s++) {
        effRed += lightSources[s].color.red * (ss[accessIdx + s]) / pixelScore;
        effBlue += lightSources[s].color.blue * (ss[accessIdx + s]) / pixelScore;
        effGreen += lightSources[s].color.green * (ss[accessIdx + s]) / pixelScore;
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

// takes in <color.png> <traced.png> and produces an <output.png>
int main(int argc, char** argv) {
  if (argc != 5) {
    printf("Usage: raytrace <(color)*.png> <(traced)*.png> <(lights)*.txt> <(output)*.png>");
    return 1;
  } 

  const float defaultDistPixel = 1.0f;

  std::vector<lightray> lightSources;
  loadFromFile(argv[3], lightSources);
  
  png::rgb_pixel oneLight = png::rgb_pixel(255, 0, 0);
    
  png::image<png::rgb_pixel> colorImg(argv[1]);
  png::image<png::gray_pixel> tracedImg(argv[2]);
  int rows = colorImg.get_height();
  int cols = colorImg.get_width();

  if (rows != tracedImg.get_height() && cols != tracedImg.get_width()) {
    printf("Error: Color Image and Traced Image are not of same dimension\n");
    return 2;
  }

  int numSources = lightSources.size();

  int dataPieces = rows * cols * numSources;
  std::vector<float> singleScores(dataPieces, 0);

  Timer rayTracerTimer;
  
  for (int s = 0; s < numSources; s++) { // cycle through the sources
    lightray lightSource = lightSources[s];
    fillPartialScores(singleScores, s, lightSource, rows, cols, numSources, tracedImg);
  }

  // accumulate the contributions from each partial score onto the main image
  colorOutput(colorImg, singleScores, lightSources, rows, cols, numSources);

  double rayTracerTime = rayTracerTimer.elapsed();
  printf("total raytracer time: %.6fs\n", rayTracerTime);

  colorImg.write(argv[4]); // finished product
  return 0;
}