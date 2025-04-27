#include <gtest/gtest.h>

#include "opencl_manager.hpp"
#include "processors/grayscale_processor.hpp"

#include <vector>
#include <stdexcept>

// Test fixture for GrayscaleProcessor
class GrayscaleProcessorTest : public ::testing::Test {
  protected:
    void SetUp() override {
        // Initialize OpenCLManager
        manager = std::make_unique<OpenCLManager>();

        // Create a 10x10 test image (4 colors: red, green, blue, white)
        width = 10;
        height = 10;
        test_image.resize(width * height);
        for (uint32_t y = 0; y < height; ++y) {
            for (uint32_t x = 0; x < width; ++x) {
                size_t idx = y * width + x;
                if (x < 5 && y < 5) { // Top-left: Red
                    test_image[idx] = { 255, 0, 0, 255 };
                } else if (x >= 5 && y < 5) { // Top-right: Green
                    test_image[idx] = { 0, 255, 0, 255 };
                } else if (x < 5 && y >= 5) { // Bottom-left: Blue
                    test_image[idx] = { 0, 0, 255, 255 };
                } else { // Bottom-right: White
                    test_image[idx] = { 255, 255, 255, 255 };
                }
            }
        }
    }

    std::unique_ptr<OpenCLManager> manager;
    std::vector<cl_uchar4> test_image;
    uint32_t width, height;
};

TEST_F(GrayscaleProcessorTest, ProcessGrayscale) {
    // Initialize processor
    GrayscaleProcessor processor(*manager);

    // Process image
    auto output = processor.process(test_image, width, height, width, height);

    // Verify output
    ASSERT_EQ(output.size(), width * height) << "Output array size mismatch";

    // Check grayscale values (using luminance: 0.299R + 0.587G + 0.114B)
    for (uint32_t y = 0; y < height; ++y) {
        for (uint32_t x = 0; x < width; ++x) {
            size_t idx = y * width + x;
            cl_uchar4 pixel = output[idx];

            // RGB channels should be equal (grayscale)
            EXPECT_EQ(pixel.s[0], pixel.s[1]) << "R != G at (" << x << "," << y << ")";
            EXPECT_EQ(pixel.s[1], pixel.s[2]) << "G != B at (" << x << "," << y << ")";

            // Check expected grayscale value
            cl_uchar4 input_pixel = test_image[idx];
            float luminance = 0.299f * input_pixel.s[0] + 0.587f * input_pixel.s[1] + 0.114f * input_pixel.s[2];
            int expected = static_cast<int>(luminance + 0.5f);
            EXPECT_NEAR(pixel.s[0], expected, 1) << "Incorrect grayscale value at (" << x << "," << y << ")";

            // Alpha should be preserved
            EXPECT_EQ(pixel.s[3], input_pixel.s[3]) << "Alpha changed at (" << x << "," << y << ")";
        }
    }

    // Optionally save output for visual inspection
    try {
        writeImageArray("out/test_grayscale_output.png", output, width, height);
    } catch (const std::exception &e) {
        std::cerr << "Warning: Failed to save grayscale output: " << e.what() << std::endl;
    }
}