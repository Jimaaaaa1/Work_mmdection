#ifndef MYTRANSFORM_H
#define MYTRANSFORM_H
#include "myonnx.h"
#include "utils.h"
#include <onnxruntime_cxx_api.h>
#include <opencv2/core.hpp>
#include <vector>

void LetterBox(
    const cv::Mat &image, 
    cv::Mat &outImage, 
    cv::Vec4d &params, 
    const cv::Size &newShape, 
    bool autoShape = false, 
    bool scaleFill = false, 
    bool scaleUp = true, 
    int stride = 32, 
    const cv::Scalar &color = {114, 114, 114});

std::vector<std::vector<Box>> PostProcess(
    const cv::Mat &input, Ort::Value &pred, 
    Ort::Value &proto, float conf_th = 0.25,
    float iou_th=0.45,
    POSTPROCESS ptype=POSTPROCESS_YOLOV5);
#endif // MYTRANSFORM_H