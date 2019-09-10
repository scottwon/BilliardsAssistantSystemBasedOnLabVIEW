#project name
PROJECT(Billiards)

cmake_minimum_required(VERSION 2.8)

find_package( OpenCV PATHS "/usr/local/include/opencv4")
find_package(OpenGL REQUIRED)
find_package(GLUT REQUIRED) 

if(COMMAND cmake_policy)
  cmake_policy(SET CMP0003 NEW)
endif(COMMAND cmake_policy)

SET(BUILD_SHARED_LIBS ON)

## where are user-specific cmake modules
SET(CMAKE_MODULE_PATH $ENV{CMAKE_MODULE_PATH})

SET(OpenCV_INCLUDE_DIRS "/usr/local/include/opencv4")
SET(OpenCV_LINK_DIRECTORIES "/usr/local/lib")
include_directories(.)
include_directories(library ${OPENGL_INCLUDE_DIRS} ${GLUT_INCLUDE_DIRS})

SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -O3") #-fno-omit-frame-pointer")
SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -std=c++11 -O3") #-fno-omit-frame-pointer")

# source files of library "LineMatchingLib" to be created

ADD_EXECUTABLE(Billiards Billiards.cpp)
TARGET_LINK_LIBRARIES(Billiards ${OpenCV_LIBS} ${OPENGL_LIBRARIES} ${GLUT_glut_LIBRARY})

