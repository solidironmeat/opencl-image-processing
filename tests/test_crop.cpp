// tests/test_crop.cpp
#include <gtest/gtest.h>

#include "processors/crop_processor.hpp"
#include "opencl_manager.hpp"

TEST(CropProcessorTest, ValidCrop) {
    OpenCLManager manager;
    CropProcessor cropper(manager);
    std::vector<cl_uchar4> input(4 * 4, { 255, 0, 0, 255 }); // 4x4 red image

    auto output = cropper.process(input, 4, 4, 2, 2, 1, 1);

    ASSERT_EQ(output.size(), 4); // 2x2 output
    for (const auto &pixel : output) {
        EXPECT_EQ(pixel.x, 255); // Red channel
        EXPECT_EQ(pixel.y, 0);   // Green channel
        EXPECT_EQ(pixel.z, 0);   // Blue channel
    }
}

TEST(CropProcessorTest, InputImage) {
    OpenCLManager manager;
    CropProcessor cropper(manager);
    std::string input_name = "resources/input.png";
    auto [in_width, in_height] = getImageSize(input_name);
    std::vector<cl_uchar4> input = readImageArray(input_name);
    uint32_t out_width = 31;
    uint32_t out_height = 31;

    auto cropped = cropper.process(input, in_width, in_height, out_width, out_height, 371, 291);

    ASSERT_EQ(cropped.size(), out_width * out_height); // 2x2 output

    std::string output_name = "out/test_cropped.png";
    writeImageArray(output_name, cropped, out_width, out_height);

    auto exp_cropped = readImageArray(output_name);
    auto [exp_out_width, exp_out_height] = getImageSize(output_name);

    ASSERT_EQ(out_width, exp_out_width);
    ASSERT_EQ(out_height, exp_out_width);

    for (int i = 0; i < 4; i++) {
        cl_uchar4 pixel = cropped[i];
        cl_uchar4 exp_pixel = exp_cropped[i];

        EXPECT_EQ(pixel.x, exp_pixel.x); // Red channel
        EXPECT_EQ(pixel.y, exp_pixel.y); // Green channel
        EXPECT_EQ(pixel.z, exp_pixel.z); // Blue channel
        EXPECT_EQ(pixel.w, exp_pixel.w); // Alpha channel
    }
}