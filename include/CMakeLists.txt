# Create header-only library
add_library(TinyLA INTERFACE)
add_library(TinyLA::TinyLA ALIAS TinyLA)

# Set include directories for the library
target_include_directories(TinyLA INTERFACE 
    ${CMAKE_CURRENT_SOURCE_DIR}
)

# Optionally enable CUDA support
find_package(CUDAToolkit QUIET)

if (CUDAToolkit_FOUND)
    message(STATUS "CUDA support enabled - CUDA Toolkit found")
    target_compile_definitions(TinyLA INTERFACE "ENABLE_CUDA_SUPPORT")
    target_link_libraries(TinyLA INTERFACE CUDA::cudart)
else()
    message(STATUS "CUDA support disabled - CUDA Toolkit not found")
endif()