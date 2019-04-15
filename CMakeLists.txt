cmake_minimum_required(VERSION 3.0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx2 -mpclmul -mbmi")

set(SOURCE_FILES
        src/main.cpp
        src/mercuryparser.cpp
        src/tests.cpp
        src/utils.cpp
        )

add_executable(main ${SOURCE_FILES})
