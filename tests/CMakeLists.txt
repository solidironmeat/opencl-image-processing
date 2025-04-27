cmake_minimum_required(VERSION 3.10)

enable_testing()

find_package(GTest QUIET)

if (GTest_FOUND)
    add_executable(run_tests
        test_crop.cpp
        test_grayscale.cpp
        test_halftone.cpp
        # Add other test files
        ../src/opencl_manager.cpp
        ../src/image_processor.cpp
        ../src/processors/crop_processor.cpp
        ../src/processors/grayscale_processor.cpp
        ../src/processors/halftone_processor.cpp
    )

    target_include_directories(run_tests PRIVATE
        ${GTEST_INCLUDE_DIRS}
        ../include
    )

    target_link_libraries(run_tests
        OpenCL::OpenCL
        OpenImageIO::OpenImageIO
        GTest::GTest
        GTest::Main
    )

    if (OpenImageIO_FOUND)
        target_link_libraries(run_tests ${OpenImageIO_LIBRARIES})
    endif()

    add_test(NAME UnitTests COMMAND run_tests)
endif()