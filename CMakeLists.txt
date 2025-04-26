cmake_minimum_required(VERSION 3.10)
project(ImageProcessingOpenCL LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find OpenCL
find_package(OpenCL REQUIRED)
find_package(OpenImageIO REQUIRED)

message(STATUS "OpenImageIO_FOUND: ${OpenImageIO_FOUND}")
message(STATUS "OpenImageIO_INCLUDE_DIRS: ${OpenImageIO_INCLUDE_DIRS}")
message(STATUS "OpenImageIO_LIBRARIES: ${OpenImageIO_LIBRARIES}")

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
target_include_directories(image_processing PUBLIC
    /usr/local/cuda/include
    ${CMAKE_SOURCE_DIR}/include
    ${OpenCL_INCLUDE_DIRS}
    ${OpenImageIO_INCLUDE_DIRS}
)

# Link libraries
target_link_libraries(image_processing PRIVATE
    OpenCL::OpenCL
    OpenImageIO::OpenImageIO
)

# Copy kernels to build directory
file(COPY ${CMAKE_SOURCE_DIR}/kernels DESTINATION ${CMAKE_BINARY_DIR})

# Enable testing
enable_testing()
find_package(GTest QUIET)
if (GTest_FOUND)
    include_directories(${GTEST_INCLUDE_DIRS})
    add_executable(run_tests
        tests/test_crop.cpp
        tests/test_grayscale.cpp
        tests/test_halftone.cpp
        # Add other test files
        src/opencl_manager.cpp
        src/image_processor.cpp
        src/processors/crop_processor.cpp
        src/processors/grayscale_processor.cpp
        src/processors/halftone_processor.cpp
    )
    target_include_directories(run_tests PUBLIC
        ${CMAKE_SOURCE_DIR}/include
        ${OpenCL_INCLUDE_DIRS}
    )
    target_link_libraries(run_tests 
        ${OpenCL_LIBRARIES}
        GTest::GTest
        GTest::Main
    )
    if (OpenImageIO_FOUND)
        target_link_libraries(run_tests OpenCL::OpenCL OpenImageIO::OpenImageIO)
    endif()
    add_test(NAME UnitTests COMMAND run_tests)
endif()

# Installation rules (optional)
install(TARGETS image_processing DESTINATION bin)
install(DIRECTORY kernels DESTINATION share/image_processing)