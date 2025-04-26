#ifndef OPENCL_MANAGER_HPP
#define OPENCL_MANAGER_HPP

#define CL_TARGET_OPENCL_VERSION 200
#define CL_HPP_TARGET_OPENCL_VERSION 200

#include <CL/opencl.hpp>

#include <string>
#include <vector>
#include <fstream>
#include <iostream>

class OpenCLManager {
  public:
    OpenCLManager();
    cl::Context &getContext();
    cl::CommandQueue &getQueue();
    cl::Device &getDevice();

  private:
    cl::Platform platform;
    std::vector<cl::Device> devices;
    cl::Device device;
    cl::Context context;
    cl::CommandQueue queue;
};

std::string loadKernelSource(const std::string &path);
std::vector<cl_uchar4> readImageArray(const std::string &file_name);
std::pair<uint32_t, uint32_t> getImageSize(const std::string &file_name);

#endif // OPENCL_MANAGER_HPP