#include "myonnx.h"
#include "mytransform.h"
#include <numeric>
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <assert.h>
#ifdef _WIN32
#include <Windows.h>
#else
#include <sys/time.h>
#endif
#include "utils.h"

unsigned int now(){
#ifdef _WIN32
    return GetTickCount();
#else
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec * 1000 + tv.tv_usec / 1000;
#endif
}

std::vector<Ort::Value> MyONNX::one(const MyTensor &t)
{
    unsigned int t0, t1;

    t0 = now();
    std::vector<Ort::Value> input_tensors;
    input_tensors.emplace_back(Ort::Value::CreateTensor<float>(
       _OrtMemoryInfo,t.ptr,t.size,t.shape.data(),t.shape.size()
    ));
    t1 = now();
    // std::cout << "Input tensors created: "<< t1 - t0 << " ms." << std::endl;
    input_tensors[0].At<float>({0, ishape.at(1) - 1, ishape.at(2) - 1, ishape.at(3) - 1});


    t0 = now();
    std::vector<Ort::Value> output_tensors;
    output_tensors = _OrtSession->Run(Ort::RunOptions{ nullptr },
                                    input_node_names.data(),
                                    input_tensors.data(),
                                    input_tensors.size(),
                                    output_node_names.data(),
                                    output_node_names.size()); 
    t1 = now();
    // std::cout << "Infer finished: " << t1 - t0 << " ms." << std::endl;
    return output_tensors;
}

cv::Mat MyONNX::postProcess(const cv::Mat &input, std::vector<Ort::Value> &output_tensors) const
{
    if (input.empty()) return input;
    if (ptype == POSTPROCESS_NONE) return input;
    auto result = PostProcess(
        input, output_tensors[0], output_tensors[1], conf_th, iou_th, ptype);
    auto m = input.clone();
    for (auto &b : result[0])
        b.applyMask(m);
    return m;
}

void printModelInfo(Ort::Session &session, Ort::AllocatorWithDefaultOptions &allocator)
{
    using namespace std;
    // 输出模型输入节点的数量
    size_t num_input_nodes = session.GetInputCount();
    size_t num_output_nodes = session.GetOutputCount();
    cout << "Number of input node is:" << num_input_nodes << endl;
    cout << "Number of output node is:" << num_output_nodes << endl;

    // 获取输入输出维度
    for (decltype(num_input_nodes) i = 0; i < num_input_nodes; i++)
    {
        std::vector<int64_t> input_dims = session.GetInputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape();
        cout << "input " << i << " dim is: ";
        for (decltype(input_dims.size()) j = 0; j < input_dims.size(); j++)
            cout << input_dims[j] << " ";
        cout << endl;
    }
    for (decltype(num_output_nodes) i = 0; i < num_output_nodes; i++)
    {
        std::vector<int64_t> output_dims = session.GetOutputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape();
        cout << "output " << i << " dim is: ";
        for (decltype(output_dims.size()) j = 0; j < output_dims.size(); j++)
            cout << output_dims[j] << " ";
        cout << endl;
    }
    // 输入输出的节点名
    cout << endl; // 换行输出
    for (decltype(num_input_nodes) i = 0; i < num_input_nodes; i++)
    #ifdef _WIN32
        cout << "The input op-name " << i << " is: " << session.GetInputNameAllocated(i, allocator).get() << endl;
    #else
        cout << "The input op-name " << i << " is: " << session.GetInputName(i, allocator) << endl;
    #endif
    for (decltype(num_output_nodes) i = 0; i < num_output_nodes; i++)
    #ifdef _WIN32
        cout << "The output op-name " << i << " is: " << session.GetOutputNameAllocated(i, allocator).get() << endl;
    #else
        cout << "The output op-name " << i << " is: " << session.GetOutputName(i, allocator) << endl;
    #endif
}

MyTensor MyONNX::read(cv::Mat & img) const
{
    auto t0 = now();
    auto tmp = cv::Vec4d{1, 1, 0, 0};
    auto size = cv::Size{
        static_cast<int>(ishape.at(3)),
        static_cast<int>(ishape.at(2))};
    LetterBox(img, img, tmp, size);
    auto t1 = now();
    // std::cout << "LetterBox: " << t1 - t0 << " ms." << img.size() << std::endl;

    t0 = now();
    auto ret = LoadImageToPtr(img);
    t1 = now();
    // std::cout << "HWC->CHW: " << t1 - t0 << " ms." << std::endl;
    return ret;
}

bool MyONNX::load(const std::string &modelPath)
{
    using namespace std;
    try
    {
        std::vector<std::string> available_providers = Ort::GetAvailableProviders();
        auto cuda_available = std::find(available_providers.begin(), available_providers.end(), "CUDAExecutionProvider");
        #ifndef _WIN32
            OrtCUDAProviderOptions provider_options;
            _OrtSessionOptions.AppendExecutionProvider_CUDA(provider_options);
        #endif
        if (_OrtSession != nullptr) _OrtSession->release();
        #ifdef _WIN32
        std::wstring model_ws(modelPath.begin(), modelPath.end());
        _OrtSession = new Ort::Session(_OrtEnv, model_ws.c_str(), _OrtSessionOptions);
        #else
        _OrtSession = new Ort::Session(_OrtEnv, modelPath.c_str(), _OrtSessionOptions);
        #endif
        printModelInfo(*_OrtSession, _OrtAllocator);

        ishape = _OrtSession->GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
        ptype = POSTPROCESS_NONE;
        auto oshape = _OrtSession->GetOutputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
        if (oshape.at(1) < oshape.at(2)) { // TODO: fixme
            ptype = POSTPROCESS_YOLOV8;
            std::cout << "POSTPROCESS_YOLOV8" << std::endl;
        } else {
            ptype = POSTPROCESS_YOLOV5;
            std::cout << "POSTPROCESS_YOLOV5" << std::endl;
        }
        return true;
    }
    catch (const std::exception &e)
    {
        cout << (e.what()) << endl;
        _OrtSession = nullptr;
        return false;
    }
}

MyTensor LoadImageToPtr(const cv::Mat &img, float scale)
{
    auto w = img.cols;
    auto h = img.rows;
    auto ch = img.channels();
    decltype(img.ptr<uchar>()) ptr;

    auto data = new float[ch * h * w];

    for (int i = 0; i < h; ++i)
    {
        ptr = img.ptr<uchar>(i);
        for (int j = 0; j < w; ++j)
        {
            for (int c = 0; c < ch; ++c)
            {
                // data: c h w
                data[c * (h * w) +
                     i * w +
                     j] = ptr[j * ch + c] * scale;
                // ptr: h w c
            }
        }
    }
    auto shape = std::array<int64_t, 4>{1, ch, h, w};
    auto ret = MyTensor{};
    ret.ptr = data;
    ret.size = ch * h * w;
    ret.shape = shape;
    return ret;
}
