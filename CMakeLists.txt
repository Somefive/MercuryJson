cmake_minimum_required(VERSION 3.0)

set(CMAKE_CXX_STANDARD 17)
if (CMAKE_BUILD_TYPE MATCHES "Debug")
    set(CMAKE_CXX_FLAGS_DEBUG "-g -O0")
    message("Debug mode")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_DEBUG}")
else ()
    set(CMAKE_CXX_FLAGS_RELEASE "-O3")
    message("Release mode")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_RELEASE}")
endif ()

if (CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    # clang does not support -mavx2 and stuff
    message("Using ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native -pthread")
else ()
    message("Using non-Clang compiler ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx2 -mpclmul -mbmi -mbmi2 -march=native -pthread")
endif ()

message("Compile flags: ${CMAKE_CXX_FLAGS}")

set(SOURCE_FILES
        src/find_structurals.cpp
        src/mercuryparser.cpp
        src/recursive_descent_parser.cpp
        src/shift_reduce_parser.cpp
        src/tape.cpp
#        src/tests.cpp
        src/utils.cpp
        src/parsestring.cpp
        )

add_executable(main src/main.cpp ${SOURCE_FILES})
add_executable(benchmark_parsestring src/benchmark/parsestring ${SOURCE_FILES})

include(benchmark/CMakeLists.txt)

