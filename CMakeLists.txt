CMAKE_MINIMUM_REQUIRED(VERSION 3.10)

PROJECT(RayTracing)

SET(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/build/bin)

SET(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/build/lib)

SET(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} "-Wall -O -g -std=c++17 -pthread")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

ADD_DEFINITIONS(-DLOCAL_LOG_DIR=\"${PROJECT_SOURCE_DIR}/Log/\")

#########################################################
# Find OpenCV
#########################################################

# FIND_PACKAGE(OpenCV REQUIRED)
# INCLUDE_DIRECTORIES(${OpenCV_INCLUDE_DIRS})


# MESSAGE(STATUS "OpenCV library status:")
# MESSAGE(STATUS "---- version: ${OpenCV_VERSION}")
# MESSAGE(STATUS "---- libraries: ${OpenCV_LIBS}")
# MESSAGE(STATUS "---- include path: ${OpenCV_INCLUDE_DIRS}")

# #########################################################
# # Find OpenCL
# #########################################################
FIND_PACKAGE(OpenCL REQUIRED)
INCLUDE_DIRECTORIES(${OpenCL_INCLUDE_DIRS})

MESSAGE(STATUS "OpenCL library status:")
MESSAGE(STATUS "---- libraries: ${OpenCL_LIBRARY}")
MESSAGE(STATUS "---- include path: ${OpenCL_INCLUDE_DIRS}")

#########################################################

INCLUDE_DIRECTORIES(src/Framework/include)
ADD_SUBDIRECTORY(src/Framework)

INCLUDE_DIRECTORIES(ext)
INCLUDE_DIRECTORIES(ext/imgui/include)
INCLUDE_DIRECTORIES(ext/glad/include)
INCLUDE_DIRECTORIES(ext/GLFW/include)
INCLUDE_DIRECTORIES(ext/stb)


ADD_SUBDIRECTORY(ext/imgui)
ADD_SUBDIRECTORY(ext/glad)
ADD_SUBDIRECTORY(ext/GLFW)


LINK_LIBRARIES(Framework)
LINK_LIBRARIES(glfw)
LINK_LIBRARIES(imgui)
LINK_LIBRARIES(glad)

ADD_EXECUTABLE(opengl main_opengl.cpp)