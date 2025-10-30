# Create header-only library
add_library(TinyLA INTERFACE)
add_library(TinyLA::TinyLA ALIAS TinyLA)

# Set include directories for the library
target_include_directories(TinyLA INTERFACE 
    ${CMAKE_CURRENT_SOURCE_DIR}
)

# Optionally enable CUDA support
if (TRUE)
    message(STATUS "CUDA support enabled")
    target_compile_definitions(TinyLA INTERFACE "ENABLE_CUDA_SUPPORT")

    find_package(CUDAToolkit REQUIRED)
    target_link_libraries(TinyLA INTERFACE CUDA::cudart)
endif()