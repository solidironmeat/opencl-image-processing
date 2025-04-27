#ifndef IMAGE_PROCESSOR_HPP
#define IMAGE_PROCESSOR_HPP

#include "opencl_manager.hpp"

class ImageProcessor {
  public:
    ImageProcessor(OpenCLManager &manager, const std::string &kernelSource, const std::string &kernelName);

    virtual std::vector<cl_uchar4> process(const std::vector<cl_uchar4> &input, uint32_t in_width,
                                           uint32_t in_height,                      //
                                           uint32_t out_width, uint32_t out_height, //
                                           uint32_t in_start_x = 0, uint32_t in_start_y = 0) = 0;

  protected:
    OpenCLManager &manager;
    cl::Program program;
    cl::Kernel kernel;
};

#endif // IMAGE_PROCESSOR_HPP