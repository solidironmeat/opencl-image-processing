#ifndef CROP_PROCESSOR_HPP
#define CROP_PROCESSOR_HPP

#include "../image_processor.hpp"

class CropProcessor : public ImageProcessor {
  public:
    CropProcessor(OpenCLManager &manager, uint32_t start_x = 1, uint32_t start_y = 1);

    std::vector<cl_uchar4> process(const std::vector<cl_uchar4> &input_array, uint32_t in_width, uint32_t in_height,
                                   uint32_t out_width, uint32_t out_height) override;

  private:
    static const char *cropKernelSource;
    uint32_t start_x;
    uint32_t start_y;
};

#endif // CROP_PROCESSOR_HPP