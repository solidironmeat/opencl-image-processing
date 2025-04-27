#include "image_processor.hpp"

ImageProcessor::ImageProcessor(OpenCLManager &manager, const std::string &kernelSource, const std::string &kernelName)
    : manager(manager) {
    // Create and build program
    cl::Program::Sources sources;
    sources.push_back({ kernelSource.c_str(), kernelSource.size() });
    program = cl::Program(manager.getContext(), sources);
    try {
        program.build({ manager.getDevice() });
    } catch (...) {
        std::string log = program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(manager.getDevice());
        throw std::runtime_error("Failed to build OpenCL program: " + std::string("Build log: " + log));
    }

    // Create kernel
    cl_int err;
    kernel = cl::Kernel(program, kernelName.c_str(), &err);
    if (err != CL_SUCCESS) {
        throw std::runtime_error("Failed to create kernel: error " + std::to_string(err));
    }
}