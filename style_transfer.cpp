#define USE_DML

#include <assert.h>
#include <stdio.h>
#include <unordered_map>
#include <string>
#include <onnxruntime_cxx_api.h>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <chrono>


constexpr int64 IMAGE_HEIGHT = 720;
constexpr int64 IMAGE_WIDTH = 720;
constexpr int IMAGE_CHANNELS = 3;

void loadImage(const std::string& filename, int sizeX, int sizeY, std::vector<float>& inputImage, int& orgWidth, int& orgHieght);


/**
 * convert input from CHW format to HWC format
 * \param input A single image. This float array has length of 3*h*w
 * \param h image height
 * \param w image width
 * \param output A byte array. should be freed by caller after use
 */
static void chw_to_hwc(const float* input,const size_t h, const size_t w, uint8_t* output) {
    size_t stride = h * w;

    for (size_t c = 0; c != 3; ++c) {
        size_t t = c * stride;
        for (size_t i = 0; i != stride; ++i) {
            float f = input[t + i];
            if (f < 0.f)
                f = 0;
            else if(f > 255.0f)
                f = 255.f;
            output[i * 3 + c] = (uint8_t)f;
        }
    }
}

static void usage() { std::cout << "usage: <model_path> <input_file> <output_file> [cpu|cuda|dml]" << std::endl; }
static int run_inference(Ort::Session* session, Ort::RunOptions* runOptions, std::string const& input_file,
    std::string const& output_file) {

    constexpr int64_t numInputElements = IMAGE_HEIGHT * IMAGE_WIDTH * IMAGE_CHANNELS;

    const std::array<int64_t, 4> inputShape = { 1, IMAGE_CHANNELS, IMAGE_HEIGHT, IMAGE_WIDTH };
    const std::array<int64_t, 4> outputShape = { 1, IMAGE_CHANNELS, IMAGE_HEIGHT, IMAGE_WIDTH };

    std::vector<float> imageVec(numInputElements);
    std::vector<float> inputImage(numInputElements);
    int orgW , orgH;

    // load image
    loadImage(input_file, IMAGE_WIDTH, IMAGE_HEIGHT, imageVec, orgW, orgH);
    if (imageVec.empty()) {
        std::cout << "Failed to load image: " << input_file << std::endl;
        return 1;
    }

    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);



    auto inputTensor = Ort::Value::CreateTensor<float>(memory_info, imageVec.data(), imageVec.size(), inputShape.data(), inputShape.size());
    
    std::vector<float> output(numInputElements);
    auto outputTensor = Ort::Value::CreateTensor<float>(memory_info, output.data(), output.size(), outputShape.data(),
        outputShape.size());

    assert(inputTensor != nullptr);
    assert(inputTensor.IsTensor());

    // define names
    Ort::AllocatorWithDefaultOptions ort_alloc;
    Ort::AllocatedStringPtr inputName = session->GetInputNameAllocated(0, ort_alloc);
    Ort::AllocatedStringPtr outputName = session->GetOutputNameAllocated(0, ort_alloc);
    const std::array<const char*, 1> inputNames = { inputName.get() };
    const std::array<const char*, 1> outputNames = { outputName.get() };
    inputName.release();
    outputName.release();

    using std::chrono::high_resolution_clock;
    using std::chrono::duration_cast;
    using std::chrono::duration;
    using std::chrono::milliseconds;
    auto t1 = high_resolution_clock::now();

    // run inference
    try {
        session->Run(*runOptions, inputNames.data(), &inputTensor, 1, outputNames.data(), &outputTensor, 1);
    }
    catch (Ort::Exception& e) {
        std::cout << e.what() << std::endl;
        return 1;
    }

    auto t2 = high_resolution_clock::now();
    /* Getting number of milliseconds as an integer. */
    auto ms_int = duration_cast<milliseconds>(t2 - t1);
    std::cout<<" time to run:"<< ms_int.count() <<std::endl;

    assert(outputTensor != nullptr);

    assert(outputTensor.IsTensor());
    int ret = 0;

    std::vector<uint8_t> output_image_data(IMAGE_HEIGHT * IMAGE_WIDTH * IMAGE_CHANNELS);
    chw_to_hwc(output.data(), IMAGE_HEIGHT, IMAGE_WIDTH, &output_image_data[0]);

    cv::Mat imgbuf = cv::Mat(IMAGE_HEIGHT, IMAGE_WIDTH, CV_8UC3, &output_image_data[0], IMAGE_HEIGHT * IMAGE_CHANNELS);
    //cv::cvtColor(imgbuf, imgbuf, cv::COLOR_BGR2RGB);
    cv::resize(imgbuf, imgbuf, cv::Size(orgW,orgH));

    cv::imshow("output", imgbuf);
    try {
        cv::imwrite(output_file, imgbuf);
    }
    catch (const cv::Exception& ex) {
        std::cout << ex.what() << std::endl;
    }
    cv::waitKey();
    return ret;

}

static void verify_input_output_count(Ort::Session* session) {
    assert(session->GetInputCount() == 1);
    assert(session->GetOutputCount() == 1);
}
/*
int enable_cuda(Ort::SessionOptions* session_options) {
  // OrtCUDAProviderOptions is a C struct. C programming language doesn't have constructors/destructors.
  OrtCUDAProviderOptions o;
  // Here we use memset to initialize every field of the above data struct to zero.
  memset(&o, 0, sizeof(o));
  // But is zero a valid value for every variable? Not quite. It is not guaranteed. In the other words: does every enum
  // type contain zero? The following line can be omitted because EXHAUSTIVE is mapped to zero in onnxruntime_c_api.h.
  o.cudnn_conv_algo_search = OrtCudnnConvAlgoSearchExhaustive;
  o.gpu_mem_limit = SIZE_MAX;
  OrtStatus* onnx_status = g_ort->SessionOptionsAppendExecutionProvider_CUDA(session_options, &o);
  if (onnx_status != NULL) {
    const char* msg = g_ort->GetErrorMessage(onnx_status);
    fprintf(stderr, "%s\n", msg);
    g_ort->ReleaseStatus(onnx_status);
    return -1;
  }
  return 0;
}
*/


cv::Mat hwc2chw(const cv::Mat& image) {
    std::vector<cv::Mat> rgb_images;
    cv::split(image, rgb_images);

    // Stretch one-channel images to vector
    cv::Mat m_flat_r = rgb_images[0].reshape(1, 1);
    cv::Mat m_flat_g = rgb_images[1].reshape(1, 1);
    cv::Mat m_flat_b = rgb_images[2].reshape(1, 1);

    // Now we can rearrange channels if need
    cv::Mat matArray[] = { m_flat_r, m_flat_g, m_flat_b };

    cv::Mat flat_image;
    // Concatenate three vectors to one
    cv::hconcat(matArray, 3, flat_image);
    return flat_image;
}

#ifdef _WIN32
int wmain(int argc, wchar_t* argv[]) {
#else
int main(int argc, char* argv[]) {
#endif
    if (argc < 4) {
        usage();
        return -1;
    }
    std::wstring model_path = std::wstring(argv[1]);
    std::wstring winput_file = std::wstring(argv[2]);
    std::wstring woutput_file = std::wstring(argv[3]);

    std::string output_file(woutput_file.begin(), woutput_file.end());
    std::string input_file(winput_file.begin(), winput_file.end());
    // By default it will try CUDA first. If CUDA is not available, it will run all the things on CPU.
    // But you can also explicitly set it to DML(directml) or CPU(which means cpu-only).
    std::wstring execution_provider = (argc >= 5) ? std::wstring(argv[4]) : L"";

    Ort::Env env;
    Ort::RunOptions runOptions;

    Ort::SessionOptions ort_session_options;

    if (!execution_provider.empty()) {
        if (execution_provider.compare(L"cpu") == 0) {
            // Nothing; this is the default
        }
        else if (execution_provider.compare(L"dml") == 0) {
#ifdef USE_DML
            ort_session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
            std::unordered_map<std::string, std::string> dml_options;
            dml_options["performance_preference"] = "high_performance";
            dml_options["device_filter"] = "gpu";
            dml_options["disable_metacommands"] = "false";
            dml_options["enable_graph_capture"] = "false";

            ort_session_options.AppendExecutionProvider("DML", dml_options);
#else
            std::cout << "DirectML is not enabled in this build." << std::endl;
            return -1;
#endif
        }
        else if (execution_provider.compare(L"cuda") == 0) {
            std::cout << "Try to enable CUDA first" << std::endl;
            int ret = 1;  // enable_cuda(session_options);
            if (ret) {
                std::cout << "CUDA is not available" << std::endl;
                return -1;
            }
            else {
                std::cout << "CUDA is enabled" << std::endl;
            }
        }
    }

    Ort::Session session{ env, model_path.c_str(), ort_session_options };

    // run inference 
    int runRet = run_inference(&session, &runOptions, input_file, output_file);

    verify_input_output_count(&session);

    if (runRet != 0) {
        std::cout << "Failed to run inference" << std::endl;
    }
    return runRet;
}


void loadImage(const std::string& filename, int sizeX, int sizeY, std::vector<float>& inputImage, int& orgWidth, int& orgHieght) {
    cv::Mat image = cv::imread(filename);
    std::vector<float> vec;

    if (image.empty()) {
        std::cout << "No image found.";
        return ;
    }

    // model accepts BGR, no need to convert

    orgWidth = image.size().width;
    orgHieght = image.size().height;
    // resize
    cv::resize(image, image, cv::Size(sizeX, sizeY), cv::InterpolationFlags::INTER_AREA);

    //for debugging
    //cv::imshow("test", image);
    //cv::waitKey();

    std::vector<cv::Mat> rgb_images;
    cv::split(image, rgb_images);

    // convert to chw
    // Stretch one-channel images to vector
    cv::Mat m_flat_r = rgb_images[0].reshape(1, 1);
    cv::Mat m_flat_g = rgb_images[1].reshape(1, 1);
    cv::Mat m_flat_b = rgb_images[2].reshape(1, 1);

    // Now we can rearrange channels if need
    cv::Mat matArray[] = { m_flat_r, m_flat_g, m_flat_b };

    cv::Mat flat_image;
    // Concatenate three vectors to one
    cv::hconcat(matArray, 3, flat_image);

    flat_image.convertTo(inputImage, CV_32FC1);
}