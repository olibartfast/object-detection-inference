#pragma once
#include "common.hpp"

class InferenceEngine{
    	
    public:
        InferenceEngine(const std::string& modelConfiguration, 
         const std::string& modelBinary,
         bool use_gpu = false)
        {

        }



        static void SetLogger(const std::shared_ptr<spdlog::logger>& logger) 
        {
            logger_ = logger;
        }

        
        virtual std::tuple<std::vector<std::vector<float>>, std::vector<std::vector<int64_t>>> get_infer_results(const cv::Mat& input_blob) = 0;

    protected:

        static std::shared_ptr<spdlog::logger> logger_; 



};