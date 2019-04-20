cmake_minimum_required(VERSION 3.0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -mavx2 -mpclmul -mbmi -mbmi2")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g")

set(SOURCE_FILES
        src/main.cpp
        src/mercuryparser.cpp
        src/tests.cpp
        src/utils.cpp
        )

add_executable(main ${SOURCE_FILES})
