How to install x86-64 variant of homebrew:
 (This link should be all the info you need)
 https://gist.github.com/progrium/b286cd8c82ce0825b2eb3b0b3a0720a0


 How to get packages once you have x86-64 variant of homebrew installed: 
 https://stackoverflow.com/questions/72970363/how-do-you-install-the-x86-64-version-of-a-brew-package-on-a-m1-mac


 After getting homebrew, download the module libpng. This would be done via `arch -x86_64 /usr/local/bin/brew install libpng` from the command line

 Then you will need to install png++ which uses libpng (which was originally made for standard C to deal with image processing files) but png++ makes use of those nice quick syntaxes to simplify our convolution.

 This may work installing for png++:
 `brew install png++`

 If everything is done correctly, you should be able to compile the `testRay.cpp` file in the repo via `g++ -lpng16 testRay.cpp`


 To get ISPC running: `brew install ispc`


 Compile `convolution.cpp` file via `g++ -lpng16 convolution.cpp` and then run `./a.out <png_file>`. The output will be stored in images/outputConv.png.
 ---------------------------------------------------------------------------------------------------------------
 Updates 11/27:
 M1 Macs have a hard time using OpenMP due to the ARM processor
 So instead of using the regular `brew` all installations should use the `arch -x86_64 /usr/local/bin/brew` version

 The compiler being used should no longer be the default `g++` version. Rather, first install via 
 `arch -x86_64 /usr/local/bin/brew install gcc@12`. The compiler we will be using when this is done can be called via `x86_64-apple-darwin21-g++-12` as seen in the makefile. Then do another brew install for openMP via
 `arch -x86_64 /usr/local/bin/brew install libomp`. Stuff should work then, but if openMP does not work you may need to install `llvm` via this similar process.

 Important: Make sure when using the new compiler that you are linking the correct installs of `png++` and `libpng`, for the x86_64 brew installation. Linking these can be rather tricky but specifying a `-I/{path to library}` flag during compilation can account for this.