// tests/test_crop.cpp
#include <gtest/gtest.h>

#include "processors/crop_processor.hpp"
#include "opencl_manager.hpp"

TEST(CropProcessorTest, ValidCrop) {
    OpenCLManager manager;
    CropProcessor cropper(manager, 1, 1);
    std::vector<cl_uchar4> input(4 * 4, { 255, 0, 0, 255 }); // 4x4 red image
    auto output = cropper.process(input, 4, 4, 2, 2);
    ASSERT_EQ(output.size(), 4); // 2x2 output
    for (const auto &pixel : output) {
        EXPECT_EQ(pixel.x, 255); // Red channel
        EXPECT_EQ(pixel.y, 0);   // Green channel
        EXPECT_EQ(pixel.z, 0);   // Blue channel
    }
}