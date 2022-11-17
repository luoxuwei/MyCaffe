cmake_minimum_required(VERSION 3.10.2)

project("MyCaffe")
include_directories(.)

add_library(mycaffe-core SHARED
        net.h
        layer.h
        net.cpp)

find_library(
        armadillo-lib
        armadillo
)
find_library(
        jsoncpp-lib
        jsoncpp
)
find_library(
        protobuf-lib
        protobuf
)

target_link_libraries(
        mycaffe-core
        ${armadillo-lib}
        ${jsoncpp-lib}
        ${protobuf-lib}
)

add_executable(MyCaffe main.cpp blob.h blob.cpp)

target_link_libraries(MyCaffe mycaffe-core)

