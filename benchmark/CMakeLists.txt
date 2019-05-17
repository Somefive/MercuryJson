cmake_minimum_required(VERSION 3.0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O0")
if (CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    # clang does not support -mavx2 and stuff
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")
else ()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx2 -mpclmul -mbmi -mbmi2")
endif ()

set(SUPPORT_FILES
        src/utils.cpp
        )

add_executable(rapidjson benchmark/rapidjson.cpp ${SUPPORT_FILES})
set_target_properties(rapidjson PROPERTIES EXCLUDE_FROM_ALL 1)
