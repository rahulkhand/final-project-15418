#include <png.h>
#include <stdio.h>
#include <stdlib.h>
#include <zlib.h>
#include <string>
#include <png++/png.hpp>

#define MARGIN 1

bool similarColors(int c1, int c2) {
    return abs(c1 - c2) <= MARGIN;
}

// program runs via ./imageCompare <image1> <image2>
int main(int argc, char** argv) {
    png::image<png::rgb_pixel> image1(argv[1]);
    png::image<png::rgb_pixel> image2(argv[2]);

    int rows = image1.get_height();
    int cols = image1.get_width();

    for (int r = 0; r < rows; r++) {
        for (int c = 0; c < cols; c++) {
            if (!(similarColors(image1[r][c].red, image2[r][c].red) && 
                similarColors(image1[r][c].green, image2[r][c].green) && 
                similarColors(image1[r][c].blue, image2[r][c].blue))) {
                printf("Images Differ at row: %d, col: %d\n", r, c);
                printf("Image 1 pixel: [%d, %d, %d]\n", (int)image1[r][c].red, (int)image1[r][c].green, (int)image1[r][c].blue);
                printf("Image 2 pixel: [%d, %d, %d]\n", (int)image2[r][c].red, (int)image2[r][c].green, (int)image2[r][c].blue);
                return 1;
            }
        }
    }

    printf("Images are similar!\n");
    return 0;
}