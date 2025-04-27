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
        std::vector<cl_uchar4> input = readImageArray(input_name);
        std::filesystem::create_directories("resources");
        uint32_t out_width = 170;
        uint32_t out_height = 170;

        OpenCLManager manager;
        CropProcessor cropper(manager);
        auto cropped = cropper.process(input, in_width, in_height, out_width, out_height, 232, 316);
        writeImageArray("resources/cropped.png", cropped, out_width, out_height);

        GrayscaleProcessor grayscaler(manager);
        auto grayed = grayscaler.process(cropped, out_width, out_height, out_width, out_height);
        writeImageArray("resources/grayed.png", grayed, out_width, out_height);

        HalftoneProcessor halftoner(manager);
        auto halftoned = halftoner.process(grayed, out_width, out_height, out_width, out_height);
        writeImageArray("resources/halftoned.png", halftoned, out_width, out_height);
    } catch (const std::exception &e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    } catch (...) {
        std::cerr << "Unknown error occurred" << std::endl;
        return 1;
    }
    return 0;
}