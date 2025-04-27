#include "opencl_manager.hpp"
#include "processors/crop_processor.hpp"
#include "processors/grayscale_processor.hpp"
#include "processors/halftone_processor.hpp"

int main(int argc, char *argv[]) {
    try {
        if (argc != 2) {
            throw std::runtime_error("\nUsage: ./image_processing <input_file>");
        }
        std::string input_name = argv[1];
        auto [in_width, in_height] = getImageSize(input_name);
        std::vector<cl_uchar4> input_array = readImageArray(input_name);
        std::filesystem::create_directories("resources");
        OpenCLManager manager;
        uint32_t out_width = 200, out_height = 200;

        CropProcessor cropper(manager);
        auto cropped = cropper.process(input_array, in_width, in_height, out_width, out_height, 50, 50);
        writeImageArray("resources/cropped.png", cropped, out_width, out_height);

        GrayscaleProcessor grayscaler(manager);
        auto grayed = grayscaler.process(input_array, in_width, in_height, in_width, in_height);
        writeImageArray("resources/grayed.png", grayed, in_width, in_height);

        HalftoneProcessor halftoner(manager);
        auto halftoned = halftoner.process(grayed, in_width, in_height, in_width, in_height);
        writeImageArray("resources/halftoned.png", halftoned, in_width, in_height);
    } catch (const std::exception &e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    } catch (...) {
        std::cerr << "Unknown error occurred" << std::endl;
        return 1;
    }
    return 0;
}