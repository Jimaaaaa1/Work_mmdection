#include "utils.h"
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

Box::Box(int i, float conf, int x1, int y1, int x2, int y2) : id(i), conf(conf), x1(x1), y1(y1), x2(x2), y2(y2)
{
}

float Box::iou(const Box &other) const
{
    auto max_x = std::max(x1, other.x1);
    auto min_x = std::min(x2, other.x2);
    auto max_y = std::max(y1, other.y1);
    auto min_y = std::min(y2, other.y2);
    if (min_x <= max_x || min_y <= max_y)
        return 0;
    float over_area = (float)(min_x - max_x) * (min_y - max_y); // 计算重叠面积
    float area_a = area();
    float area_b = other.area();
    float iou = over_area / (area_a + area_b - over_area);
    return iou;
}

float Box::area() const
{
    return static_cast<float>((x2 - x1) * (y2 - y1));
}

cv::Rect2i Box::rect() const
{
    return cv::Rect2i{x1, y1, x2 - x1, y2 - y1};
}

void Box::applyMask(cv::Mat im) const
{
    if (im.empty())
        return;
    cv::Mat tmp = im.clone();
    cv::Mat resizedMask;
    cv::resize(mask, resizedMask, im.size());
    tmp(rect()).setTo(cv::Scalar{
                          0, //static_cast<float>(rand() % 256),
                          0, //static_cast<float>(rand() % 256),
                          255 //static_cast<float>(rand() % 256)
                          },
                      resizedMask(rect()) > 0.5);
    cv::addWeighted(im, 0.5, tmp, 0.5, 0, im);
    cv::rectangle(im, rect(), {0, 0, 255});
    cv::putText(im, std::to_string(conf), {x1, y1}, cv::FONT_HERSHEY_COMPLEX_SMALL, 1, {0, 0, 255});
}

void nms_iou(std::vector<Box> &d, float iou_th, int maxdet)
{
    using namespace std;
    std::sort(d.begin(), d.end(), [](const Box &l, const Box &r)
              { return l.conf > r.conf; });
    auto n = d.size();
    for (int i = 0; i < n; ++i)
    {
        for (int j = 0; j < i; ++j)
        {
            if (d[j].valid && d[i].iou(d[j]) >= iou_th)
            {
                d[i].valid = false;
                break;
            }
        }
    }

    d.erase(remove_if(d.begin(), d.end(),
                      [](const Box &b)
                      { return !b.valid; }),
            d.end());
    if (d.size() > maxdet) d.erase(d.begin() + maxdet, d.end());
    std::sort(d.begin(), d.end(), [](const Box &l, const Box &r)
            { return l.id < r.id; });
}
