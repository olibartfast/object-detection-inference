#pragma once
#include "InferenceInterface.hpp"
#include <NvInfer.h>  // for TensorRT API
#include <cuda_runtime_api.h>  // for CUDA runtime API
#include <fstream>

#include "Logger.hpp"

class TRTInfer : public InferenceInterface
{
    protected:
        std::shared_ptr<nvinfer1::ICudaEngine> engine_{nullptr};
        nvinfer1::IExecutionContext* context_{nullptr};
        std::vector<void*> buffers_;
        std::vector<std::vector<int64_t>> output_shapes_;
        std::vector<std::vector<float>> h_outputs_;
        nvinfer1::IRuntime* runtime_{nullptr};

    public:
        TRTInfer(const std::string& model_path);

        // Create execution context and allocate input/output buffers
        void createContextAndAllocateBuffers();

        void initializeBuffers(const std::string& engine_path);

        // calculate size of tensor
        size_t getSizeByDim(const nvinfer1::Dims& dims);    

        void infer();

        std::tuple<std::vector<std::vector<float>>, std::vector<std::vector<int64_t>>> get_infer_results(const cv::Mat& input_blob) override;

        ~TRTInfer()
        {
            for (void* buffer : buffers_)
            {
                cudaFree(buffer);
            }
        }


};