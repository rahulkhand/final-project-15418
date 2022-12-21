#include <stdio.h>
__global__ void kernel(int *x, int r, int c, int p, int *re) {
    if (threadIdx.x > 20) return;
    atomicAdd(x, 2);
    __syncthreads();
    *x = *x + 1;
    //printf("%d\n", *x);
    *re = *x;
    return;
}

int main() {
    int r = 4; 
    int c = 5; 
    int p = 6;

    const int tpb = 512;
    int blocks = (r*c*p + tpb - 1) / tpb;
    printf("blocks : %d \n", blocks);

    int A[4][5][6];
    //int B[4][5];
    for (int i = 0; i < r; i++) {
        for (int j = 0; j < c; j++) {
            for (int k = 0; k < p; k++) {
                A[i][j][k] = i + 1;
            }
        }
    }

    int *bird = (int*)malloc(sizeof(int));
    *bird = 5;

    int *mem;
    int *rest;
    cudaMalloc(&mem, sizeof(int));
    cudaMalloc(&rest, sizeof(int));
    cudaMemcpy(mem, bird, 1, cudaMemcpyHostToDevice);
    kernel<<<blocks, tpb>>>(mem, r, c, p, rest);
    cudaDeviceSynchronize();
    cudaMemcpy(bird, rest, 1, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    printf("aha: %d\n", *bird);
    free(bird);
    cudaFree(mem);
    return 0;
}