main:
	g++ -o dist/main -mavx2 -mpclmul -mbmi src/main.cpp src/utils.cpp src/mercuryparser.hpp src/tests.hpp