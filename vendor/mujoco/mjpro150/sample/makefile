COMMON=-O2 -I../include -L../bin -std=c++11 -lstdc++ -mavx

all:
	clang $(COMMON) basic.cpp      -lmujoco150 -lglfw.3 -o ../bin/basic
	clang $(COMMON) simulate.cpp   -lmujoco150 -lglfw.3 -o ../bin/simulate
	clang $(COMMON) test.cpp       -lmujoco150          -o ../bin/test
	clang $(COMMON) compile.cpp    -lmujoco150          -o ../bin/compile
	clang $(COMMON) derivative.cpp -lmujoco150          -o ../bin/derivative
	clang $(COMMON) record.cpp     -lmujoco150 -lglfw.3 -o ../bin/record
	