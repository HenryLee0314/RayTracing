
INCLUDE_DIRECTORIES(.)
INCLUDE_DIRECTORIES(include)
INCLUDE_DIRECTORIES(${PROJECT_SOURCE_DIR}/ext)
INCLUDE_DIRECTORIES(${PROJECT_SOURCE_DIR}/ext/imgui/include)
INCLUDE_DIRECTORIES(${PROJECT_SOURCE_DIR}/src/CodeAnalyzer/include)
INCLUDE_DIRECTORIES(${OpenCV_INCLUDE_DIRS})

add_library(Framework
    log.cpp
    opencl_manager.cpp
    opencl_task.cpp
)

target_link_libraries(Framework ${OpenCL_LIBRARY})