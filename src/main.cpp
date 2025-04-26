#include "opencl_manager.hpp"
#include "processors/crop_processor.hpp"
#include "processors/grayscale_processor.hpp"
#include "processors/halftone_processor.hpp"

int main(int argc, char *argv[]) {
    try {
        if (argc != 3) {
            throw std::runtime_error("Usage: " + std::string(argv[0]) + " <input_file> <output_file>");
        }

        OpenCLManager manager;

        // Input image data
        std::string file_name = "resources/input.png";
        auto [in_width, in_height] = getImageSize(file_name);
        std::vector<cl_uchar4> input_array = readImageArray(file_name);

        // Crop
        CropProcessor cropper(manager, 100, 100);
        auto cropped = cropper.process(input_array, in_width, in_height, 1280, 720);

        // Grayscale
        GrayscaleProcessor grayscaler(manager);
        auto grayed = grayscaler.process(cropped, 1280, 720, 1280, 720);

        // Halftone
        HalftoneProcessor halftoner(manager);
        auto halftoned = halftoner.process(grayed, 1280, 720, 1280, 720);

        // Use halftoned output
    } catch (const std::exception &e) {
        std::cerr << "Error: " << e.what() << std::endl;
    }
    return 0;
}