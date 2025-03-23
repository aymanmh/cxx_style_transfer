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

std::vector<float> loadImage(const std::string& filename, int sizeX, int sizeY);

/**
 * convert input from HWC format to CHW format
 * \param input A single image. The byte array has length of 3*h*w
 * \param h image height
 * \param w image width
 * \param output A float array. should be freed by caller after use
 * \param output_count Array length of the `output` param
 */

void hwc_to_chw(const uint8_t* input, size_t h, size_t w, float** output, size_t* output_count) {
    size_t stride = h * w;
    *output_count = stride * 3;
    float* output_data = (float*)malloc(*output_count * sizeof(float));
    assert(output_data != NULL);
    for (size_t i = 0; i != stride; ++i) {
        for (size_t c = 0; c != 3; ++c) {
            output_data[c * stride + i] = input[i * 3 + c];
        }
    }
    *output = output_data;
}

/**
 * convert input from CHW format to HWC format
 * \param input A single image. This float array has length of 3*h*w
 * \param h image height
 * \param w image width
 * \param output A byte array. should be freed by caller after use
 */
static void chw_to_hwc(const float* input, size_t h, size_t w, uint8_t* output) {
    size_t stride = h * w;

    for (size_t c = 0; c != 3; ++c) {
        size_t t = c * stride;
        for (size_t i = 0; i != stride; ++i) {
            float f = input[t + i];
            if (f < 0.f || f > 255.0f) f = 0;
            output[i * 3 + c] = (uint8_t)f;
        }
    }
}

static void usage() { std::cout << "usage: <model_path> <input_file> <output_file> [cpu|cuda|dml]" << std::endl; }
int run_inference(Ort::Session* session, Ort::RunOptions* runOptions, std::string const& input_file,
    std::string const& output_file) {
    size_t input_height;
    size_t input_width;
    float* model_input;
    size_t model_input_ele_count;

    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);

    constexpr int64_t numChannels = 3;
    constexpr int64_t imageWidth = 720;
    constexpr int64_t imageHeight = 720;
    constexpr int64_t numInputElements = numChannels * imageWidth * imageHeight;

    const std::array<int64_t, 4> inputShape = { 1, numChannels, imageHeight, imageWidth };
    const std::array<int64_t, 4> outputShape = { 1, numChannels, imageHeight, imageWidth };

    // define array
    std::vector<float> input(numInputElements);
    std::vector<float> output(numInputElements);

    const int64_t input_shape[] = { 1, 3, 720, 720 };


    auto inputTensor = Ort::Value::CreateTensor<float>(memory_info, input.data(), input.size(), inputShape.data(), inputShape.size());
    auto outputTensor = Ort::Value::CreateTensor<float>(memory_info, output.data(), output.size(), outputShape.data(),
        outputShape.size());

    // load image
    const std::vector<float> imageVec = loadImage(input_file, 720, 720);
    if (imageVec.empty()) {
        std::cout << "Failed to load image: " << input_file << std::endl;
        return 1;
    }

    std::copy(imageVec.begin(), imageVec.end(), input.begin());

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

    // run inference
    try {
        session->Run(*runOptions, inputNames.data(), &inputTensor, 1, outputNames.data(), &outputTensor, 1);
    }
    catch (Ort::Exception& e) {
        std::cout << e.what() << std::endl;
        return 1;
    }

    assert(outputTensor != nullptr);

    assert(outputTensor.IsTensor());
    int ret = 0;
    float* output_tensor_data = NULL;

    std::vector<uint8_t> output_image_data(720 * 720 * 3);
    chw_to_hwc(outputTensor.GetTensorMutableData<float>(), 720, 720, &output_image_data[0]);

    cv::Mat imgbuf = cv::Mat(720, 720, CV_8UC3, &output_image_data[0], 720 * 3);
    cv::imshow("output", imgbuf);
    cv::imwrite(output_file, imgbuf);

    cv::waitKey();
    return ret;

}

void verify_input_output_count(Ort::Session* session) {
    size_t count;
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


std::vector<float> loadImage(const std::string& filename, int sizeX, int sizeY) {
    cv::Mat image = cv::imread(filename);
    if (image.empty()) {
        std::cout << "No image found.";
        //return;
    }

    // convert from BGR to RGB
   // cv::cvtColor(image, image, cv::COLOR_BGR2RGB);

    // resize
    // Wait for a keystroke.
    cv::resize(image, image, cv::Size(sizeX, sizeY));


    std::vector<cv::Mat> rgb_images;
    cv::split(image, rgb_images);

    // Stretch one-channel images to vector
    cv::Mat m_flat_b = rgb_images[0].reshape(1, 1);
    cv::Mat m_flat_g = rgb_images[1].reshape(1, 1);
    cv::Mat m_flat_r = rgb_images[2].reshape(1, 1);

    // Now we can rearrange channels if need
    cv::Mat matArray[] = { m_flat_b, m_flat_g, m_flat_r };

    cv::Mat flat_image;
    // Concatenate three vectors to one
    cv::hconcat(matArray, 3, flat_image);

    std::vector<float> vec;
    flat_image.convertTo(vec, CV_32FC1);
    return vec;
}