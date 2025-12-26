cmake_minimum_required(VERSION 3.13)
project(pinn-flow-cylinder LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Warnings
if (MSVC)
  add_compile_options(/W4)
else()
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Sources
set(SOURCES
    src/matrix.cpp
    src/dense_layer.cpp
    src/pinn_model.cpp
    src/flow_field.cpp
    src/main.cpp
)

# Headers
include_directories(${CMAKE_SOURCE_DIR}/include)

add_executable(pinn_flow_cylinder ${SOURCES})

# On Unix-like systems, ensure high-precision math is linked if needed
# target_link_libraries(pinn_flow_cylinder m)
