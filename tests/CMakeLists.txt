cmake_minimum_required(VERSION 3.10)

# Enable testing
enable_testing()
find_package(GTest QUIET)
if (GTest_FOUND)
    include_directories(${GTEST_INCLUDE_DIRS})
    add_executable(run_tests
        test_crop.cpp
        test_grayscale.cpp
        test_halftone.cpp
        # Add other test files
        opencl_manager.cpp
        image_processor.cpp
        processors/crop_processor.cpp
        processors/grayscale_processor.cpp
        processors/halftone_processor.cpp
    )
    target_link_libraries(run_tests ${OpenCL_LIBRARIES} GTest::GTest GTest::Main)
    if (OpenImageIO_FOUND)
        target_link_libraries(run_tests ${OpenImageIO_LIBRARIES})
    endif()
    add_test(NAME UnitTests COMMAND run_tests)
endif()