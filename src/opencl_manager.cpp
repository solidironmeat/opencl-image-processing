#include "opencl_manager.hpp"

#include <OpenImageIO/imageio.h>

OpenCLManager::OpenCLManager() {
    // Initialize OpenCL
    platform = cl::Platform::getDefault();
    platform.getDevices(CL_DEVICE_TYPE_ALL, &devices);
    if (devices.empty()) {
        throw std::runtime_error("No OpenCL devices found");
    }
    device = devices[0];
    context = cl::Context(device);
    queue = cl::CommandQueue(context, device);
}

cl::Context &OpenCLManager::getContext() {
    return context;
}

cl::CommandQueue &OpenCLManager::getQueue() {
    return queue;
}

cl::Device &OpenCLManager::getDevice() {
    return device;
}

// Utility function to load kernel source
std::string loadKernelSource(const std::string &path) {
    std::ifstream file(path);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open kernel file: " + path);
    }
    return std::string((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
}

std::vector<cl_uchar4> readImageArray(const std::string &file_name) {
    using namespace OIIO;
    auto inp = ImageInput::open(file_name);
    if (!inp) {
        throw std::runtime_error("Failed to load image: " + file_name + " (" + geterror() + ")");
    }

    const ImageSpec &spec = inp->spec();
    int width = spec.width;
    int height = spec.height;
    std::vector<unsigned char> pixels(width * height * 4); // RGBA
    inp->read_image(TypeDesc::UINT8, pixels.data());
    inp->close();

    std::vector<cl_uchar4> image_array(width * height);
    for (int i = 0; i < width * height; ++i) {
        image_array[i] = {
            pixels[i * 4 + 0], // R
            pixels[i * 4 + 1], // G
            pixels[i * 4 + 2], // B
            pixels[i * 4 + 3]  // A
        };
    }

    return image_array;
}

std::pair<uint32_t, uint32_t> getImageSize(const std::string &file_name) {
    using namespace OIIO;
    auto inp = ImageInput::open(file_name);
    if (!inp) {
        throw std::runtime_error("Failed to get image size: " + file_name + " (" + geterror() + ")");
    }

    const ImageSpec &spec = inp->spec();
    uint32_t width = spec.width;
    uint32_t height = spec.height;
    inp->close();
    return { width, height };
}

void writeImageArray(const std::string &file_name, const std::vector<cl_uchar4> &image_array, uint32_t width,
                     uint32_t height) {
    // Validate inputs
    if (image_array.size() < width * height) {
        throw std::runtime_error("Image array size is too small for specified dimensions: "
                                 + std::to_string(image_array.size()) + " < " + std::to_string(width * height));
    }
    if (width == 0 || height == 0) {
        throw std::runtime_error("Invalid image dimensions: " + std::to_string(width) + "x" + std::to_string(height));
    }

    // Create output image
    auto out = OIIO::ImageOutput::create(file_name);
    if (!out) {
        throw std::runtime_error("Failed to create image output: " + file_name + " (" + OIIO::geterror() + ")");
    }

    // Define image specification (RGBA, 8-bit per channel)
    OIIO::ImageSpec spec(width, height, 4, OIIO::TypeDesc::UINT8);

    // Convert cl_uchar4 to unsigned char array
    std::vector<unsigned char> pixels(width * height * 4);
    for (size_t i = 0; i < width * height; ++i) {
        pixels[i * 4 + 0] = image_array[i].s[0]; // R
        pixels[i * 4 + 1] = image_array[i].s[1]; // G
        pixels[i * 4 + 2] = image_array[i].s[2]; // B
        pixels[i * 4 + 3] = image_array[i].s[3]; // A
    }

    // Write image
    if (!out->open(file_name, spec)) {
        throw std::runtime_error("Failed to open output image: " + file_name + " (" + OIIO::geterror() + ")");
    }
    if (!out->write_image(OIIO::TypeDesc::UINT8, pixels.data())) {
        throw std::runtime_error("Failed to write image: " + file_name + " (" + OIIO::geterror() + ")");
    }
    out->close();
}