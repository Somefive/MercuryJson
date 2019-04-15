main:
	g++ -std=c++17 -mavx2 -mpclmul -mbmi src/main.cpp src/utils.cpp src/tests.cpp src/mercuryparser.cpp -o dist/main
