#include <onnxruntime_cxx_api.h>
#include "base.hpp"
#include "yolo.hpp"
#include "CLI11.hpp"
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <filesystem>
#include "tracker.hpp"
#include <thread>
#include <mutex>
#include <condition_variable>
// #define NORENDER
#ifndef NORENDER
std::condition_variable cond{};
std::mutex mu{};
cv::Mat image_to_show;
void show_worker(){
    static cv::Mat thisimg;
    while (true){
        std::unique_lock<std::mutex> lk(mu);
        cond.wait(lk);
        thisimg = std::move(image_to_show);
        lk.unlock();
        if (thisimg.empty()) continue;
        cv::imshow("result", thisimg);
        cv::waitKey(1);
    }
}
#endif

int main(int argc, char **argv)
{
    CLI::App app{"YOLOv5-DeepSort"};
    std::string model_path = "yolov5n.onnx";
    std::string image_path = "imgs/demo5.jpg";
    std::string output_path = "imgs/demo5_out.jpg";
    std::string video = "main.avi";
    float nms_th = 0.45;
    float conf_th = 0.25;
    int max_det = 20;
    float k = 1e-3;
    bool v8 = false;
    float lost = 0.0;
    app.add_option("-m,--model", model_path, "Path to model");
    app.add_option("-i,--image", image_path, "Path to image");
    app.add_option("-o,--output", output_path, "Path to output image");
    app.add_option("-c,--conf", conf_th, "Confidence threshold");
    app.add_option("-t,--nms", nms_th, "NMS threshold");
    app.add_option("-d,--maxdet", max_det, "Max detections");
    app.add_option("-v,--video", video, "Path to video");
    app.add_flag("-8", v8, "yolov8 checkpoint");
    app.add_option("-k", k, "Kalman filter noise");
    app.add_option("-l", lost, "Lost threshold(probability)");
    CLI11_PARSE(app, argc, argv);

    Ort::YOLO yolo(model_path.c_str());
    if (v8)
        yolo.v5 = false;

    AllTrackers trackers;
    trackers.states = 10;
    trackers.noise = k;

    if (video == "")
    {
        cv::Mat image = cv::imread(image_path);
        auto boxes = yolo.detect(image, conf_th, nms_th, max_det);
        std::cout << "Found " << boxes.size() << " boxes" << std::endl;
        trackers.update(boxes);
        trackers.render(image);
        cv::imwrite(output_path, image);
        return 0;
    }
    #ifdef _WIN32
    if (std::filesystem::status(video).type() == std::filesystem::file_type::directory)
    #else
    if (false)
    #endif
    {
        for (int i = 4; i <= 7000; i = (i + 1) % 7001)
        {
            std::string image_path = video + "/" + std::to_string(i) + ".jpg";
            cv::Mat image = cv::imread(image_path);
            if (image.empty())
                continue;
            auto boxes = yolo.detect(image, conf_th, nms_th, max_det);
            std::cout << "Found " << boxes.size() << " boxes" << std::endl;
            if (!trackers.update(boxes, lost))
            {
                for (auto &b : boxes)
                {
                    cv::putText(image, std::to_string(b.conf), b.rect().br(),
                                cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 0, 255), 2);
                    cv::rectangle(image, b.rect(), {255, 0, 0}, 2);
                }
            }
            trackers.render(image);
            cv::imshow("YOLOv5-DeepSort", image);
            if (cv::waitKey(1) == 'q')
                break;
        }
        return 0;
    }
    cv::VideoCapture cap;
    if (video.size() == 1)
    {
        cap.open(video[0] - '0');
    }
    else
    {
        cap.open(video);
    }
    if (!cap.isOpened())
    {
        std::cerr << "Cannot open video" << std::endl;
        return 1;
    }
    cv::Mat frame;
    #ifndef NORENDER
    std::thread(show_worker).detach();
    #endif
    while (cap.read(frame))
    {
        // std::cout << trackers.fps.fps(now()) << std::endl;
        // continue;
        auto boxes = yolo.detect(frame, conf_th, nms_th, max_det);
        std::cout << "Found " << boxes.size() << " boxes" << std::endl;
// #define NORENDER
#ifdef NORENDER
        // trackers.update(boxes, false);
        std::cout << trackers.fps.fps(now()) << std::endl;
#endif

#ifndef NORENDER
        if (!trackers.update(boxes, lost))
        {
            for (auto &b : boxes)
            {
                cv::putText(frame, std::to_string(b.conf), b.rect().br(),
                            cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 0, 255), 2);
                cv::rectangle(frame, b.rect(), {255, 0, 0}, 2);
            }
        }
        trackers.render(frame);
        mu.lock();
        image_to_show = std::move(frame);
        mu.unlock();
        cond.notify_one();
#endif
    }
    return 0;
}