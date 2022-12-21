CPPFLAGS=-I/usr/local/Cellar/libpng/1.6.39/include -I/usr/local/Cellar/png++/0.2.10/include -L/usr/local/Cellar/libpng/1.6.39/lib -lpng16
IMAGEDIR=../images
IMAGE=terra.png
BRIDGESFLAGS=-I/jet/home/rkhandel/include

all: rays compare

rays: rays-seq rays-par rays-par-v1

rays-ext: rays-seq-ext rays-par-ext compare-ext rays-cuda
	
rays-seq-ext:
	g++ raytracer-sequential.cpp $(BRIDGESFLAGS) `libpng-config --ldflags` -o raytrace-sequential

rays-par-ext:
	g++ raytracer-parallel.cpp $(BRIDGESFLAGS) `libpng-config --ldflags` -o raytrace-parallel -fopenmp

compare-ext:
	g++ imageCompare.cpp $(BRIDGESFLAGS) `libpng-config --ldflags` -o imageCompare

rays-seq: raytracer-sequential.cpp
	x86_64-apple-darwin21-g++-12 $(CPPFLAGS) raytracer-sequential.cpp -o raytrace-sequential

rays-par: raytracer-parallel.cpp
	x86_64-apple-darwin21-g++-12 $(CPPFLAGS) raytracer-parallel.cpp -o raytrace-parallel -fopenmp

rays-par-v1: raytracer-parallel-v1.cpp
	x86_64-apple-darwin21-g++-12 $(CPPFLAGS) raytracer-parallel-v1.cpp -o raytrace-parallel-v1 -fopenmp

# This will only work on machine with Nvida GPU
rays-cuda: raytracer.cu
	nvcc raytracer.cu $(BRIDGESFLAGS) `libpng-config --ldflags` -o raytrace-cuda

compare: imageCompare.cpp
	x86_64-apple-darwin21-g++-12 $(CPPFLAGS) imageCompare.cpp -o imageCompare

run: raytrace-sequential raytrace-parallel imageCompare
	./raytrace-sequential $(IMAGEDIR)/$(IMAGE) $(IMAGEDIR)/convolution_output.png lights.txt $(IMAGEDIR)/outputSeq.png
	./raytrace-parallel $(IMAGEDIR)/$(IMAGE) $(IMAGEDIR)/convolution_output.png lights.txt $(IMAGEDIR)/outputPar.png
	./imageCompare $(IMAGEDIR)/outputSeq.png $(IMAGEDIR)/outputPar.png

lights:
	python3 ../scripts/generateTests.py

speed:
	python3 ../scripts/speedup.py

clean:
	rm -rf raytrace-sequential raytrace-parallel rays-par-v1 raytrace-cuda imageCompare 
	rm -rf $(IMAGEDIR)/outputCuda.png $(IMAGEDIR)/outputSeq.png $(IMAGEDIR)/outputPar.png
	rm -rf lights.txt

