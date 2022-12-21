CXX86=x86_64-apple-darwin21-g++-12
CXXISPC=ispc --arch=x86_64 --colored-output
CXX=g++
IMAGE=terra.png
CXXFLAGS=-I/usr/local/Cellar/libpng/1.6.39/include -I/usr/local/Cellar/png++/0.2.10/include -L/usr/local/Cellar/libpng/1.6.39/lib -lpng16
BRIDGESFLAGS=-I/jet/home/rkhandel/include
NUMTHREADS=1 # set argument from cmdline for testing

all: convolution

convolution: convolution.cpp convolution.ispc
	ispc convolution.ispc -o convolution_ispc.o -h convolution_ispc.h
	$(CXX86) $(CXXFLAGS) convolution.cpp convolution_ispc.o -o $@ -fopenmp

convolution-ext: convolution.cpp convolution.ispc
	ispc convolution.ispc -o convolution_ispc.o -h convolution_ispc.h
	$(CXX) $< `libpng-config --ldflags` $(BRIDGESFLAGS) convolution_ispc.o -o convolution -fopenmp

runout: convolution ../images/$(IMAGE)
	./convolution ../images/$(IMAGE)

runprof: convolution ../images/$(IMAGE)
	./convolution ../images/$(IMAGE) --profile $(NUMTHREADS)

runprof-ext: convolution-ext ../images/$(IMAGE)
	./convolution ../images/$(IMAGE) --profile $(NUMTHREADS)

clean:
	rm -rf convolution
	rm -rf convolution_ispc.o
	rm -rf ../images/outputConv.png
	rm -rf ../images/convolution_output.png