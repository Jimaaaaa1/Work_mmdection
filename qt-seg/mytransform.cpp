#include "utils.h"
#include "mytransform.h"
#include <numeric>
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

void LetterBox(const cv::Mat &image, cv::Mat &outImage, cv::Vec4d &params, const cv::Size &newShape,
               bool autoShape, bool scaleFill, bool scaleUp, int stride, const cv::Scalar &color)
{
    using namespace cv;

    cv::Size shape = image.size();
    float r = std::min((float)newShape.height / (float)shape.height,
                       (float)newShape.width / (float)shape.width);
    if (!scaleUp)
        r = std::min(r, 1.0f);

    float ratio[2]{r, r};
    int new_un_pad[2] = {(int)std::round((float)shape.width * r), (int)std::round((float)shape.height * r)};

    auto dw = (float)(newShape.width - new_un_pad[0]);
    auto dh = (float)(newShape.height - new_un_pad[1]);

    if (autoShape)
    {
        dw = (float)((int)dw % stride);
        dh = (float)((int)dh % stride);
    }
    else if (scaleFill)
    {
        dw = 0.0f;
        dh = 0.0f;
        new_un_pad[0] = newShape.width;
        new_un_pad[1] = newShape.height;
        ratio[0] = (float)newShape.width / (float)shape.width;
        ratio[1] = (float)newShape.height / (float)shape.height;
    }

    dw /= 2.0f;
    dh /= 2.0f;

    if (shape.width != new_un_pad[0] && shape.height != new_un_pad[1])
    {
        resize(image, outImage, cv::Size(new_un_pad[0], new_un_pad[1]));
    }
    else
    {
        outImage = image.clone();
    }

    int top = int(std::round(dh - 0.1f));
    int bottom = int(std::round(dh + 0.1f));
    int left = int(std::round(dw - 0.1f));
    int right = int(std::round(dw + 0.1f));
    params[0] = ratio[0];
    params[1] = ratio[1];
    params[2] = left;
    params[3] = top;
    cv::copyMakeBorder(outImage, outImage, top, bottom, left, right, cv::BORDER_CONSTANT, color);
}

std::vector<std::vector<Box>> PostProcess(
    const cv::Mat &input,
    Ort::Value &pred,
    Ort::Value &proto,
    float conf_th,
    float iou_th,
    POSTPROCESS ptype)
{
    // TODO: handle classes and class confidence
    // TODO: handle batchsize > 1
    // TODO: replace with cv::dnn::NMSBoxes

    using std::cout, std::cin, std::endl, std::vector, std::exception;
    auto predShape = pred.GetTensorTypeAndShapeInfo().GetShape();  // bnc/bcn
    auto protShape = proto.GetTensorTypeAndShapeInfo().GetShape(); // b nm h1 w1
    auto batchsize = protShape.at(0);
    vector<vector<Box>> boxes(batchsize);
    int num_classes = static_cast<int>(predShape.at(1) - 4);

    for (int i = 0; i < batchsize; ++i)
    {
        auto mask_protos = cv::Mat(
            static_cast<int>(protShape.at(1)),
            static_cast<int>(protShape.at(2) * protShape.at(3)),
            CV_32FC1, (void *)proto.GetTensorData<float>());

        auto mask_weights = cv::Mat(
            static_cast<int>(predShape.at(1)),
            static_cast<int>(predShape.at(2)),
            CV_32FC1, (void *)pred.GetTensorData<float>());

        if (POSTPROCESS_YOLOV8 == ptype)
            mask_weights = mask_weights.t(); // to b n c

        auto xywh = mask_weights(cv::Rect2i(0, 0, 4, mask_weights.rows)); // pred[:, 0:0+4]

        cv::Mat conf;
        if (POSTPROCESS_YOLOV5 == ptype)
            conf = mask_weights(cv::Rect2i(4, 0, 1, mask_weights.rows)); // pred[:, 4:4+1]
        else
        {
            conf = mask_weights(
                cv::Rect2i(4, 0, mask_weights.cols - 4 - mask_protos.rows, mask_weights.rows));
            cv::reduce(conf, conf, 1, cv::REDUCE_MAX);
        }

        mask_weights = mask_weights(
            cv::Rect2i(mask_weights.cols - mask_protos.rows, 0, mask_protos.rows, mask_weights.rows));

        auto proposals = mask_weights.rows;

        // filter by confidence
        vector<int> indexes;
        for (int j = 0; j < proposals; ++j)
            if (conf.at<float>(j) >= conf_th)
                indexes.push_back(j);

        // nms
        for (auto j : indexes)
        {
            boxes[i].emplace_back(j, conf.at<float>(j),
                                  std::max(0, (int)floor(xywh.at<float>(j, 0) - xywh.at<float>(j, 2) / 2)),
                                  std::max(0, (int)floor(xywh.at<float>(j, 1) - xywh.at<float>(j, 3) / 2)),
                                  std::min(input.cols - 1, (int)ceil(xywh.at<float>(j, 0) + xywh.at<float>(j, 2) / 2)),
                                  std::min(input.rows - 1, (int)ceil(xywh.at<float>(j, 1) + xywh.at<float>(j, 3) / 2)));
        }
        nms_iou(boxes[i], iou_th);
        if (boxes[i].size() == 0) continue;

        cv::Mat weights(static_cast<int>(boxes[i].size()), mask_weights.cols, CV_32FC1);
        for (int j = 0; j < boxes[i].size(); ++ j)
            mask_weights.row(boxes[i][j].id).copyTo(weights.row(j));

        cv::Mat tmp;
        cv::exp(-weights * mask_protos, tmp);
        tmp = 1.0 / (1.0 + tmp);
        tmp = tmp.t();
        tmp = tmp.reshape(weights.rows, static_cast<int>(protShape.at(2)));
        // assert tmp.cols==static_cast<int>(protShape.at(3))
        vector<cv::Mat> cc;
        cv::split(tmp, cc);
        for (int j = 0; j < weights.rows; ++j)
            boxes[i][j].mask = cc[j];
    }
    return boxes;
}
