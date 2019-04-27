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
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")
else ()
    message("Using non-Clang compiler ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx2 -mpclmul -mbmi -mbmi2 -pthread")
endif ()

message("Compile flags: ${CMAKE_CXX_FLAGS}")

set(SOURCE_FILES
        src/main.cpp
        src/mercuryparser.cpp
        src/tests.cpp
        src/utils.cpp
        src/tape.cpp
        )

add_executable(main ${SOURCE_FILES})

include(benchmark/CMakeLists.txt)

