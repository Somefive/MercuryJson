cmake_minimum_required(VERSION 3.0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")
if (CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    # clang does not support -mavx2 and stuff
    message("Using Clang compiler")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")
else ()
    message("Using non-Clang compiler ${CMAKE_CXX_COMPILER_ID}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx2 -mpclmul -mbmi -mbmi2 -pthread")
endif ()

set(SOURCE_FILES
        src/main.cpp
        src/mercuryparser.cpp
        src/tests.cpp
        src/utils.cpp
        src/tape.cpp
        )

add_executable(main ${SOURCE_FILES})

include(benchmark/CMakeLists.txt)
