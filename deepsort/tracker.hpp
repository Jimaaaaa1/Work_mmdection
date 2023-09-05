#pragma once
#include <opencv2/video/tracking.hpp>
#include <iostream>
#include "yolo.hpp"
#include "Hungarian.h"
#include <opencv2/core.hpp>
#include <set>
#include <queue>
#include <vector>
unsigned int now();
class Tracker
{
public:
    static int global_id;
    int i = 0;
    cv::KalmanFilter KF;
    Ort::Box s;
    Tracker(const Ort::Box &b, int n = 4, float noise = 1e-3);
    int lost = 0;
    void predict();
    void update(Ort::Box box, bool reset_loss = true);
    float cost(Ort::Box box);
};

class FPS
{
public:
    std::queue<decltype(now())> times;
    const char *fps(decltype(now()) t);

private:
    char ss[32];
};

class AllTrackers
{
public:
    FPS fps;
    std::vector<Tracker> trackers;
    int states = 8;
    float noise = 1e-3;
    bool update(const std::vector<Ort::Box> &boxes, float lost = 0.0);
    void render(cv::Mat &image);
    HungarianAlgorithm hungarian;
};

std::vector<std::pair<int, int>> hungarian(const cv::Mat &costs);
