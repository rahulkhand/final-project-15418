#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <sstream>
#include <png++/png.hpp>
#include "../raytracer/vec2.h"
#include <math.h>
#include "../raytracer/common.h"

typedef std::pair<int,int> P;
P indices[9] = {P(-1, -1), P(-1, 0), P(-1, 1), P(0, -1),  P(0, 0),  P(0, 1), P(1, -1),  P(1, 0),  P(1, 1)};
int mask[3][3] = {
                {-1, -1, -1},
                {-1, 8, -1},
                {-1, -1, -1}
                };



bool rayInBox(Vec2<float> tl, Vec2<float> br, lightray mRay) {
  Vec2<float> brHead = br - (mRay.position);
  Vec2<float> tlHead = (mRay.position) - tl;
  return (tlHead.x >= 0 && tlHead.y >= 0 && brHead.x >= 0 && brHead.y >= 0);
}

bool almostEqual(float a, float b) {
  return (abs(b - a) <= 10e-12);
}

// updates the position and intensity of a ray while contributing its partial score to the new image
void updateRay(lightray *source, lightray *mRay, float *ss, int cols, int numSources, int sourceNum) {
  if (!(almostEqual(mRay->position.x, source->position.x) && almostEqual(mRay->position.y, source->position.y))) {
    int accessRow = (int)mRay->position.y;
    int accessCol = (int)mRay->position.x;

    float *accessPoint = ss + (accessRow * cols * numSources) + (accessCol * numSources) + sourceNum;
    #pragma omp atomic
    *accessPoint += mRay->intensity;
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
  if (score < L1_THRESHOLD) return 0.f;
  else if (score < L2_THRESHOLD) return tuningCnst + tuningScale * log(score);
  else return 1.f;
}

// casts rays from a single source identified by sourceNum to update singleScores
void fillPartialScores(float *singleScores, int sourceNum, lightray lightSource, int rows, int cols, int numSources,
                    png::image<png::gray_pixel> tracedImg) {
  const Vec2f tl(0.0f, 0.0f);
  const Vec2f br((float)cols, (float)rows);
  const Vec2f unitVecX(1.0f, 0.0f);

  // spawn rays from source
  #pragma omp parallel for schedule(dynamic, RAYS_PER_THREAD)
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

float sumFloats(float *fs, int n) {
  float ans = 0.f;
  for (int i = 0; i < n; i++) {
    ans += *(fs + i);
  }
  return ans;
}

void colorOutput(png::image<png::rgb_pixel> &colorImg, float* ss, std::vector<lightray> lightSources, int rows, int cols, int numSources) {
    // apply source light to image

  #pragma omp parallel for collapse(2)
  for (int r = 0; r < rows; r++) {
    for (int c = 0; c < cols; c++) {
      float *accessPoint = ss + (r * cols * numSources) + (c * numSources);
      float pixelScore = sumFloats(accessPoint, numSources);

      float effRed = 0;
      float effGreen = 0;
      float effBlue = 0;
      for (int s = 0; s < numSources; s++) {
        effRed += lightSources[s].color.red * (*(accessPoint + s)) / pixelScore;
        effBlue += lightSources[s].color.blue * (*(accessPoint + s)) / pixelScore;
        effGreen += lightSources[s].color.green * (*(accessPoint + s)) / pixelScore;
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

void convolution_openmp(int rows, int cols, png::image<png::gray_pixel> &img, png::image<png::gray_pixel> &output){
    #pragma omp parallel for collapse(2)
    for(int i = 1; i<rows-1; i++){ //edges arent computable
        for(int j = 1; j<cols-1; j++){
            int sum = 0;
            for(int loc = 0; loc < sizeof(indices)/sizeof(P); loc++){
                P diff = indices[loc];
                sum += img[i+diff.first][j+diff.second] * mask[1+diff.first][1+diff.second];
            }
            output[i][j] = sum<=60?255:0;
        }
    }
}

// Usage: fullRays <color.png> <lights.txt> <output.png>
int main(int argc, char** argv) {

    if (argc != 4) {
        printf("Usage: Usage: fullRays <color.png> <lights.txt> <output.png>\n");
        return 1;
    } 

    // Convolution
    png::image<png::gray_pixel> img(argv[1]);
    int rows = img.get_height();
    int cols = img.get_width();
    png::image<png::gray_pixel> tracedImg(cols, rows);
    convolution_openmp(rows, cols, img, tracedImg);

    //raytracer
    std::vector<lightray> lightSources;
    loadFromFile(argv[2], lightSources);

    png::image<png::rgb_pixel> colorImg(argv[1]);
    int newRows = colorImg.get_height();
    int newCols = colorImg.get_width();

    if (rows != newRows && cols != newCols) {
        printf("Error: Color Image and Traced Image are not of same dimension\n");
        return 2;
    }

    int numSources = lightSources.size();
    void* singleScores = calloc(rows * cols * numSources, sizeof(float));

    // #pragma omp parallel for // uncomment to parallelize over sources
    for (int s = 0; s < numSources; s++) { // cycle through the sources
        lightray lightSource = lightSources[s];
        fillPartialScores((float*)singleScores, s, lightSource, rows, cols, numSources, tracedImg);
    }

    // accumulate the contributions from each partial score onto the main image
    colorOutput(colorImg, (float*)singleScores, lightSources, rows, cols, numSources);

    colorImg.write(argv[3]); // finished product
    free(singleScores);
    return 0;
}
