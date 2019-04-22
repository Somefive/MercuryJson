cmake_minimum_required(VERSION 3.0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")
if (CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    # clang does not support -mavx2 and stuff
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=haswell")
else ()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx2 -mpclmul -mbmi -mbmi2")
endif ()

set(SOURCE_FILES
        src/main.cpp
        src/mercuryparser.cpp
        src/tests.cpp
        src/utils.cpp
        )

add_executable(main ${SOURCE_FILES})
