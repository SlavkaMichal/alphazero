cmake_minimum_required(VERSION 3.0)
project(mcts)
set(PYBIND11_CPP_STANDARD -std=c++14)

find_package(pybind11 REQUIRED)
find_package(GSL REQUIRED)
find_package(Threads REQUIRED)

add_definitions(-DHEUR -DDBG)
add_compile_definitions(SIZE=169 SHAPE=13)
SET(CMAKE_CXX_FLAGS "-g -fvisibility=hidden")


FILE(GLOB SOURCES ${CMAKE_CURRENT_SOURCE_DIR} *.cpp)
FILE(GLOB HEADERS ${CMAKE_CURRENT_SOURCE_DIR} *.h)
list(FILTER SOURCES EXCLUDE REGEX ".*module.cpp$")
#set(SOURCES "main.cpp")
message(STATUS ${SOURCES})
add_executable(${PROJECT_NAME} ${SOURCES} ${HEADERS})
target_link_libraries(${PROJECT_NAME} pybind11::embed)
target_link_libraries(${PROJECT_NAME} m)
target_link_libraries(${PROJECT_NAME} Threads::Threads)
target_link_libraries(${PROJECT_NAME} GSL::gsl)
