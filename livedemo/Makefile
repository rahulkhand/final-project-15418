# Will run on Rahul's machine for Demo
SRC=../images/terra.png
TXT=lights.txt 
OUT=coolimg.png

fullrays: fullRaytracer.cpp
	x86_64-apple-darwin21-g++-12 -I/usr/local/Cellar/libpng/1.6.39/include -I/usr/local/Cellar/png++/0.2.10/include -L/usr/local/Cellar/libpng/1.6.39/lib -lpng16 fullRaytracer.cpp -o fullrays

run: fullrays
	./fullrays $(SRC) $(TXT) $(OUT)

clean:
	rm -rf fullrays
	rm -rf __pycache__