all: rays-cuda

rays-cuda: raytracer.cu
	nvcc raytracer.cu `libpng-config --ldflags` -o raytracer-cuda

run:
	./raytracer-cuda ../images/bwbw.png ../images/bwbwTrace.png lights.txt ../images/outputCuda.png

clean:
	rm -rf raytracer-cuda ../images/outputCuda.png
	