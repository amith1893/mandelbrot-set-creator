OPTIONS=-lOpenCL
CXX=g++

all:mandelbrot.exe

%.exe: %.cc
	${CXX} -Wall -g $< -o $@ $(OPTIONS)

clean:
	rm -rf *.exe
