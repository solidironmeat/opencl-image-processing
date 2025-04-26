#include "processors/crop_processor.hpp"

#include <stdexcept>
#include <fstream>

CropProcessor::CropProcessor(OpenCLManager &manager, uint32_t start_x, uint32_t start_y)
    : ImageProcessor(manager, cropKernelSource, "crop"), start_x(start_x), start_y(start_y) {
}

std::vector<cl_uchar4> CropProcessor::process(const std::vector<cl_uchar4> &input_array, uint32_t in_width,
                                              uint32_t in_height, uint32_t out_width, uint32_t out_height) {
    // Validate inputs
    if (input_array.size() < in_width * in_height) {
        throw std::runtime_error("Input array size is too small");
    }
    if (start_x + out_width > in_width || start_y + out_height > in_height) {
        throw std::runtime_error("Crop region exceeds input dimensions");
    }

    // Create buffers
    cl_int err;
    cl::Buffer bufIn(manager.getContext(), CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                     in_width * in_height * sizeof(cl_uchar4), const_cast<cl_uchar4 *>(input_array.data()), &err);
    if (err != CL_SUCCESS) {
        throw std::runtime_error("Failed to create input buffer: error " + std::to_string(err));
    }

    std::vector<cl_uchar4> output_array(out_width * out_height);
    cl::Buffer bufOut(manager.getContext(), CL_MEM_WRITE_ONLY, out_width * out_height * sizeof(cl_uchar4), nullptr,
                      &err);
    if (err != CL_SUCCESS) {
        throw std::runtime_error("Failed to create output buffer: error " + std::to_string(err));
    }

    // Set kernel arguments
    kernel.setArg(0, bufIn);
    kernel.setArg(1, bufOut);
    kernel.setArg(2, in_width);
    kernel.setArg(3, out_width);
    kernel.setArg(4, out_height);
    kernel.setArg(5, start_x);
    kernel.setArg(6, start_y);

    // Execute kernel
    cl::NDRange global(out_width, out_height);
    manager.getQueue().enqueueNDRangeKernel(kernel, cl::NullRange, global, cl::NullRange);
    manager.getQueue().finish();

    // Read output
    manager.getQueue().enqueueReadBuffer(bufOut, CL_TRUE, 0, out_width * out_height * sizeof(cl_uchar4),
                                         output_array.data());
    manager.getQueue().finish();

    return output_array;
}

const char *CropProcessor::cropKernelSource = R"(
    __kernel void crop(__global const uchar4* input,
                       __global uchar4* output,
                       uint in_width,
                       uint out_width,
                       uint out_height,
                       uint start_x,
                       uint start_y) {
        uint x = get_global_id(0);
        uint y = get_global_id(1);
        if (x < out_width && y < out_height) {
            uint in_idx = (y + start_y) * in_width + (x + start_x);
            uint out_idx = y * out_width + x;
            output[out_idx] = input[in_idx];
        }
    }
)";