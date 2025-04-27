#ifndef CROP_PROCESSOR_HPP
#define CROP_PROCESSOR_HPP

#include "../image_processor.hpp"

class CropProcessor : public ImageProcessor {
  public:
    CropProcessor(OpenCLManager &manager);
    std::vector<cl_uchar4> process(const std::vector<cl_uchar4> &input_array, //
                                   uint32_t in_width, uint32_t in_height,     //
                                   uint32_t out_width, uint32_t out_height,   //
                                   uint32_t in_start_x = 0, uint32_t in_start_y = 0) override;
};

#endif // CROP_PROCESSOR_HPP