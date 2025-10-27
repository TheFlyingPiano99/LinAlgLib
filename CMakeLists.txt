cmake_minimum_required(VERSION 3.10)

project(LinAlgLibProject VERSION 1.0.0 LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Create test executable
add_executable(linalglib_test "tests/test_main.cpp")

add_subdirectory(include)

# Link the library to the test executable
target_link_libraries(linalglib_test PRIVATE LinAlgLib)

