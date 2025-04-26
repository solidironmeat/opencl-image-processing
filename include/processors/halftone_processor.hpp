#ifndef HALFTONE_PROCESSOR_HPP
#define HALFTONE_PROCESSOR_HPP

#include "../image_processor.hpp"

class HalftoneProcessor : public ImageProcessor {
  public:
    HalftoneProcessor(OpenCLManager &manager) : ImageProcessor(manager, halftoneKernelSource, "halftone") {
    }

    std::vector<cl_uchar4> process(const std::vector<cl_uchar4> &input_array, uint32_t in_width, uint32_t in_height,
                                   uint32_t out_width, uint32_t out_height) override {
        if (out_width != in_width || out_height != in_height) {
            throw std::runtime_error("Halftone output dimensions must match input");
        }
        if (input_array.size() < in_width * in_height) {
            throw std::runtime_error("Input array size is too small");
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
        kernel.setArg(3, in_height);

        // Execute kernel
        cl::NDRange global(in_width, in_height);
        manager.getQueue().enqueueNDRangeKernel(kernel, cl::NullRange, global, cl::NullRange);
        manager.getQueue().finish();

        // Read output
        manager.getQueue().enqueueReadBuffer(bufOut, CL_TRUE, 0, out_width * out_height * sizeof(cl_uchar4),
                                             output_array.data());
        manager.getQueue().finish();

        return output_array;
    }

  private:
    static const char *halftoneKernelSource;
};

const char *HalftoneProcessor::halftoneKernelSource = R"(
        __kernel void halftone(__global const uchar4* input,
                               __global uchar4* output,
                               uint width,
                               uint height) {
            uint x = get_global_id(0);
            uint y = get_global_id(1);
            if (x < width && y < height) {
                uint idx = y * width + x;
                uchar4 pixel = input[idx];
                // Convert to grayscale
                float gray = 0.299f * pixel.x + 0.587f * pixel.y + 0.114f * pixel.z;
                // Simple 2x2 threshold matrix
                const float threshold[4] = {0.25f, 0.75f, 0.50f, 1.00f};
                uint mx = x % 2;
                uint my = y % 2;
                uint m_idx = my * 2 + mx;
                uchar out_val = gray / 255.0f > threshold[m_idx] ? 255 : 0;
                output[idx] = (uchar4)(out_val, out_val, out_val, pixel.w);
            }
        }
    )";

#endif // HALFTONE_PROCESSOR_HPP