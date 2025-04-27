#include <gtest/gtest.h>

#include "opencl_manager.hpp"
#include "processors/halftone_processor.hpp"

#include <vector>
#include <stdexcept>

// Test fixture for HalftoneProcessor
class HalftoneProcessorTest : public ::testing::Test {
  protected:
    void SetUp() override {
        // Initialize OpenCLManager
        manager = std::make_unique<OpenCLManager>();

        // Create a 10x10 grayscale test image with gradient
        width = 10;
        height = 10;
        test_image.resize(width * height);
        for (uint32_t y = 0; y < height; ++y) {
            for (uint32_t x = 0; x < width; ++x) {
                size_t idx = y * width + x;
                // Gradient from black (0) to white (255)
                uint8_t value = static_cast<uint8_t>((x * 255.0f / (width - 1)) + 0.5f);
                test_image[idx] = { value, value, value, 255 };
            }
        }
    }

    std::unique_ptr<OpenCLManager> manager;
    std::vector<cl_uchar4> test_image;
    uint32_t width, height;
};

TEST_F(HalftoneProcessorTest, ProcessHalftone) {
    // Initialize processor
    HalftoneProcessor processor(*manager);

    // Process image
    auto output = processor.process(test_image, width, height, width, height);

    // Verify output
    ASSERT_EQ(output.size(), width * height) << "Output array size mismatch";

    // Check halftone effect (pixels should be near 0 or 255)
    for (uint32_t y = 0; y < height; ++y) {
        for (uint32_t x = 0; x < width; ++x) {
            size_t idx = y * width + x;
            cl_uchar4 pixel = output[idx];

            // RGB channels should be equal (grayscale-based halftone)
            EXPECT_EQ(pixel.s[0], pixel.s[1]) << "R != G at (" << x << "," << y << ")";
            EXPECT_EQ(pixel.s[1], pixel.s[2]) << "G != B at (" << x << "," << y << ")";

            // Halftone pixels should be near black (0) or white (255)
            // Allow slight variation due to dithering
            bool is_black = pixel.s[0] <= 10;
            bool is_white = pixel.s[0] >= 245;
            EXPECT_TRUE(is_black || is_white)
                << "Pixel at (" << x << "," << y << ") is not halftone (value: " << (int) pixel.s[0] << ")";

            // Alpha should be preserved
            EXPECT_EQ(pixel.s[3], test_image[idx].s[3]) << "Alpha changed at (" << x << "," << y << ")";
        }
    }

    // Optionally save output for visual inspection
    try {
        writeImageArray("out/test_halftone_output.png", output, width, height);
    } catch (const std::exception &e) {
        std::cerr << "Warning: Failed to save halftone output: " << e.what() << std::endl;
    }
}