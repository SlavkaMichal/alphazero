cmake_minimum_required(VERSION 3.0)
project(cmcts)
if (POLICY CMP0053)
	cmake_policy(SET CMP0053 OLD)
endif()

OPTION(HEUR "Compile with heuristics" OFF)
IF(HEUR)
	add_definitions(-DHEUR)
ENDIF(HEUR)
OPTION(DEBUG "Debug define" OFF)
IF(DEBUG)
	add_definitions(-DDEBUG)
ENDIF(DEBUG)
OPTION(THREADS "Define number of threads" OFF)
IF(THREADS)
	add_definitions(-DTHREADS=${THREADS})
ENDIF(THREADS)
OPTION(CUDA "Cuda capability" OFF)
IF(WITH_CUDA)
	add_definitions(-DWITH_CUDA)
ENDIF(WITH_CUDA)

set( CMAKE_VERBOSE_MAKEFILE on )

set(PYBIND11_CPP_STANDARD -std=c++14)

find_package(Threads REQUIRED)
set(CMAKE_PREFIX_PATH ${PREFIX})
find_package(pybind11 REQUIRED)
find_package(GSL REQUIRED)
set(CMAKE_PREFIX_PATH ${LOCAL_SITE_PATH}/torch)
find_package(Torch REQUIRED)

add_definitions(-fPIC)
add_definitions(-DSIZE=${SIZE} -DSHAPE=${SHAPE} -DMAJOR=${MAJOR} -DMINOR=${MINOR} -DGIT="${GIT}")
SET(CMAKE_CXX_FLAGS "-g -fvisibility=hidden")


SET(SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src)
SET(SOURCES
	${SOURCE_DIR}/cmcts.cpp
	${SOURCE_DIR}/state.cpp
	${SOURCE_DIR}/node.cpp)

include_directories(${SOURCE_DIR})
pybind11_add_module(${PROJECT_NAME} ${SOURCES} ${SOURCE_DIR}/module.cpp)
#add_executable(${PROJECT_NAME}_bin ${SOURCES} ${SOURCE_DIR}/main.cpp)

target_link_libraries(${PROJECT_NAME} PRIVATE "${TORCH_LIBRARIES}")
target_link_libraries(${PROJECT_NAME} PRIVATE m)
target_link_libraries(${PROJECT_NAME} PRIVATE Threads::Threads)
target_link_libraries(${PROJECT_NAME} PRIVATE GSL::gsl)
#target_link_libraries(${PROJECT_NAME}_bin PRIVATE pybind11::embed)
#target_link_libraries(${PROJECT_NAME}_bin PRIVATE "${TORCH_LIBRARIES}")
#target_link_libraries(${PROJECT_NAME}_bin PRIVATE m)
#target_link_libraries(${PROJECT_NAME}_bin PRIVATE Threads::Threads)
#target_link_libraries(${PROJECT_NAME}_bin PRIVATE GSL::gsl)
