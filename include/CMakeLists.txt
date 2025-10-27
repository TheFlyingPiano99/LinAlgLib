# Create header-only library
add_library(LinAlgLib INTERFACE)

# Set include directories for the library
target_include_directories(LinAlgLib INTERFACE 
    ${CMAKE_CURRENT_SOURCE_DIR}
)

# Optionally enable CUDA support
if (TRUE)
    message(STATUS "CUDA support enabled")
    target_compile_definitions(LinAlgLib INTERFACE ENABLE_CUDA_SUPPORT)

    find_package(CUDAToolkit REQUIRED)
    target_link_libraries(LinAlgLib INTERFACE CUDA::cudart)
    
endif()