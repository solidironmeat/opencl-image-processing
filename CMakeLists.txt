cmake_minimum_required(VERSION 3.10)
project(ImageProcessingOpenCL LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find OpenCL
find_package(OpenCL REQUIRED)
find_package(OpenImageIO REQUIRED)

# Source files
set(SOURCES
    src/opencl_manager.cpp
    src/image_processor.cpp
    src/processors/crop_processor.cpp
    src/processors/grayscale_processor.cpp
    src/processors/halftone_processor.cpp
    src/main.cpp
)

# Create executable
add_executable(image_processing ${SOURCES})

# Include directories
target_include_directories(image_processing PRIVATE
    ${CMAKE_SOURCE_DIR}/include
)

# Link libraries
target_link_libraries(image_processing PRIVATE
    OpenCL::OpenCL
    OpenImageIO::OpenImageIO
)

# Copy kernels to build directory
file(COPY ${CMAKE_SOURCE_DIR}/kernels DESTINATION ${CMAKE_BINARY_DIR})

# Enable testing
add_subdirectory(tests)

# Installation rules (optional)
install(TARGETS image_processing DESTINATION bin)
install(DIRECTORY kernels DESTINATION share/image_processing)