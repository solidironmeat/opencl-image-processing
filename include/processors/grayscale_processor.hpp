#ifndef GRAYSCALE_PROCESSOR_HPP
#define GRAYSCALE_PROCESSOR_HPP

#include "../image_processor.hpp"

class GrayscaleProcessor : public ImageProcessor {
  public:
    GrayscaleProcessor(OpenCLManager &manager);

    std::vector<cl_uchar4> process(const std::vector<cl_uchar4> &input_array, //
                                   uint32_t in_width, uint32_t in_height,     //
                                   uint32_t out_width, uint32_t out_height,   //
                                   uint32_t start_x = 0, uint32_t start_y = 0) override;

  private:
    static const char *grayscaleKernelSource;
};

#endif // GRAYSCALE_PROCESSOR_HPP