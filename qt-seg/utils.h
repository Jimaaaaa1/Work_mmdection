#ifndef UTILS_H
#define UTILS_H
#include <vector>
#include <opencv2/core.hpp>
struct Box
{
    int id;
    int x1, y1;
    int x2, y2; 

    int cls = -1;
    float conf = 0.0;
    bool valid = true;
    cv::Mat mask;

    Box(int, float, int, int, int, int);
    float iou(const Box & other) const;
    float area() const;
    cv::Rect2i rect() const;
    void applyMask(cv::Mat im) const;
    
};

void nms_iou(std::vector<Box> &d, float iou_th = 0.45, int maxdet=20);
#endif // UTILS_H