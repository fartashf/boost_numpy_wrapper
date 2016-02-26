cmake_minimum_required(VERSION 2.8 FATAL_ERROR)
project(gradientMex)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
 
find_package(Boost COMPONENTS python REQUIRED)
find_package(PythonLibs 2.7 REQUIRED)
find_package(BoostNumPy REQUIRED)
find_package(OpenMP)

set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS} -O2 -g")
 
include_directories(${Boost_INCLUDE_DIRS})
include_directories(${PYTHON_INCLUDE_DIRS})
include_directories(${BOOST_PYTHON_INCLUDE_DIR})
include_directories(${BOOST_NUMPY_INCLUDE_DIR})
 
add_library(gradientMex SHARED gradientMex_main.cpp)
# Ensure the output binary is called TestModule.so on Linux (not libTestModule.so)
set_target_properties(gradientMex PROPERTIES PREFIX "")
if(WIN32)
	# Ensure the output binary is called TestModule.pyd on Windows (not TestModule.dll)
	set_target_properties(gradientMex PROPERTIES SUFFIX ".pyd")
endif(WIN32)
target_link_libraries(gradientMex ${Boost_LIBRARIES} ${PYTHON_LIBRARIES} ${BOOST_NUMPY_LIBRARY})
