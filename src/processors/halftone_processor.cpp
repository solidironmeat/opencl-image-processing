#include "processors/halftone_processor.hpp"

HalftoneProcessor::HalftoneProcessor(OpenCLManager &manager)
    : ImageProcessor(manager, loadKernelSource("kernels/halftone.cl"), "halftone") {
}

std::vector<cl_uchar4> HalftoneProcessor::process(const std::vector<cl_uchar4> &input_array, //
                                                  uint32_t in_width, uint32_t in_height,     //
                                                  uint32_t out_width, uint32_t out_height,   //
                                                  uint32_t start_x, uint32_t start_y) {
    if (input_array.size() < in_width * in_height) {
        throw std::runtime_error("Input array size too small");
    }
    if (out_width != in_width || out_height != in_height) {
        throw std::runtime_error("Halftone processor requires same input/output dimensions");
    }

    cl::Buffer bufIn(manager.getContext(), CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                     in_width * in_height * sizeof(cl_uchar4), const_cast<cl_uchar4 *>(input_array.data()));
    std::vector<cl_uchar4> output_array(out_width * out_height);
    cl::Buffer bufOut(manager.getContext(), CL_MEM_WRITE_ONLY, out_width * out_height * sizeof(cl_uchar4), nullptr);

    kernel.setArg(0, bufIn);
    kernel.setArg(1, bufOut);
    kernel.setArg(2, in_width);
    kernel.setArg(3, out_width);
    kernel.setArg(4, out_height);

    cl::NDRange global(out_width, out_height);
    manager.getQueue().enqueueNDRangeKernel(kernel, cl::NullRange, global, cl::NullRange);
    manager.getQueue().finish();

    manager.getQueue().enqueueReadBuffer(bufOut, CL_TRUE, 0, out_width * out_height * sizeof(cl_uchar4),
                                         output_array.data());
    manager.getQueue().finish();

    return output_array;
}
