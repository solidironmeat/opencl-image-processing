#include <gtest/gtest.h>

#include "processors/crop_processor.hpp"
#include "opencl_manager.hpp"

TEST(ImageIOTest, ReadWrite) {
    OpenCLManager manager;
    CropProcessor cropper(manager);
    std::string input_name = "resources/input.png";
    auto [in_width, in_height] = getImageSize(input_name);
    std::vector<cl_uchar4> image = readImageArray(input_name);

    ASSERT_EQ(in_width, 640);
    ASSERT_EQ(in_height, 800);
    ASSERT_EQ(image.size(), in_width * in_height);

    std::string output_name = "out/test_output.png";
    writeImageArray(output_name, image, in_width, in_height);
    auto exp_image = readImageArray(output_name);
    auto [exp_out_width, exp_out_height] = getImageSize(output_name);

    ASSERT_EQ(in_width, exp_out_width);
    ASSERT_EQ(in_height, exp_out_height);

    for (int i = 0; i < 4; i++) {
        cl_uchar4 pixel = image[i];
        cl_uchar4 exp_pixel = exp_image[i];

        EXPECT_EQ(pixel.x, exp_pixel.x); // Red channel
        EXPECT_EQ(pixel.y, exp_pixel.y); // Green channel
        EXPECT_EQ(pixel.z, exp_pixel.z); // Blue channel
        EXPECT_EQ(pixel.w, exp_pixel.w); // Alpha channel
    }
}