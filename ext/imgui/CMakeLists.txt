cmake_minimum_required (VERSION 3.0)

INCLUDE_DIRECTORIES(.)
INCLUDE_DIRECTORIES(include)

add_definitions(-DIMGUI_IMPL_OPENGL_LOADER_GLAD)

LINK_DIRECTORIES(../../lib)

add_library (imgui
imgui.cpp
imgui_demo.cpp
imgui_draw.cpp
imgui_impl_glfw.cpp
imgui_impl_opengl3.cpp
imgui_widgets.cpp
)
