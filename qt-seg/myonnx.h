#ifndef MYONNX_H
#define MYONNX_H
#include <opencv2/core.hpp>
#include <onnxruntime_cxx_api.h>
#include <array>
enum POSTPROCESS{
    POSTPROCESS_NONE,
    POSTPROCESS_YOLOV5,
    POSTPROCESS_YOLOV8
};

unsigned int now();

struct MyTensor{
    float * ptr;
    size_t size;
    std::array<int64_t, 4> shape;
};

class MyONNX
{
public:
    MyTensor read(cv::Mat & img) const;
    bool load(const std::string& modelPath);
    std::vector<Ort::Value> one(const MyTensor &t);
    cv::Mat postProcess(const cv::Mat &, std::vector<Ort::Value> &) const;

    float conf_th=0.25;
    float iou_th=0.45;
    const int maxdet=20;
    POSTPROCESS ptype = POSTPROCESS_NONE;


    bool enableCuda = true;
    int cudaIndex = -1;
    std::vector<int64_t> ishape = {};

    std::vector<const char*> input_node_names = {"images"};
    std::vector<const char*> output_node_names = {"output0", "output1"};


    Ort::Env _OrtEnv = Ort::Env(OrtLoggingLevel::ORT_LOGGING_LEVEL_ERROR, "QtYolov5Seg");
    Ort::SessionOptions _OrtSessionOptions;
    Ort::Session *_OrtSession = nullptr;
    Ort::MemoryInfo _OrtMemoryInfo =
            Ort::MemoryInfo::CreateCpu(
                OrtArenaAllocator,
                OrtMemTypeDefault);
    Ort::AllocatorWithDefaultOptions _OrtAllocator;
};

MyTensor LoadImageToPtr(const cv::Mat &img, float scale = 1.0 / 255.0);
#endif // MYONNX_H