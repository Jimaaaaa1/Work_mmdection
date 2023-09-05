#include "controller.h"
#include <opencv2/imgproc.hpp>
// #include <QImage>
#include <iostream>
#include <QFile>
#include <onnxruntime_cxx_api.h>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
Controller::Controller(QObject *parent) : QObject(parent)
{
    inputSvc.inferSvc = &inferSvc;
    vidSvc.inputSvc = &inputSvc;
    QObject::connect(this, &Controller::sSeg,
                     &vidSvc, &VideoService::segMany);
    QObject::connect(&inputSvc, &OnnxInputService::sImg,
                     this, &Controller::showImage);
}

Controller::~Controller()
{
}

// ImageProvider::ImageProvider() :QQuickImageProvider(QQuickImageProvider::Image)
// {

// }

// QImage ImageProvider::requestImage(const QString &id, QSize *size, const QSize &requestedSize)
// {
//     qDebug() << requestedSize.width() << "," << requestedSize.height();
//     return this->img;
// }

// QPixmap ImageProvider::requestPixmap(const QString &id, QSize *size, const QSize &requestedSize)
// {
//     qDebug() << requestedSize.width() << "," << requestedSize.height();
//     return QPixmap::fromImage(this->img);
// }

void Controller::seg(const QString &name)
{
    if (inferSvc.onnx->_OrtSession == nullptr) 
    {std::cout<< "session is null, please reload model." << std::endl; return;}
    emit sSeg(name);
    emit inputSvc.start();
    emit inferSvc.start();
}

void Controller::stop()
{
    vidSvc.stopped = true;
}

void Controller::quit()
{
    stop();
    vidSvc.stop();
    inputSvc.stop();
    inferSvc.stop();
}

void Controller::setNMS(
    const QString & conf_th, 
    const QString & iou_th)
{
    inferSvc.onnx->conf_th = conf_th.toFloat();
    inferSvc.onnx->iou_th = iou_th.toFloat();
    qWarning() << "conf_th=" << inferSvc.onnx->conf_th << " "
               << "iou_th=" << inferSvc.onnx->iou_th;
}

void Controller::setModel(const QString &name)
{
    stop();
    // TODO: timeout
    while (! (
        inputSvc.inputs.empty() &&
        inferSvc.in->empty() &&
        inferSvc.out->empty()
    )) QThread::msleep(1);

    inferSvc.onnx->load(name.toStdString());
}

void Controller::showImage(const cv::Mat &i)
{
    cv::imshow("result", i);
    cv::waitKey(1);
}

OnnxInputService::OnnxInputService(QObject *parent): Daemon(parent)
{
}

OnnxInputService::~OnnxInputService()
{
}

void OnnxInputService::loop()
{
    FPS fps;
    auto  &inferSvc = *this -> inferSvc;
    qDebug("Input Service: %lld", QThread::currentThreadId());
    std::queue<cv::Mat> tmp;
#ifdef _WIN32
    auto WINNAME = "result";
    cv::namedWindow(WINNAME, cv::WINDOW_AUTOSIZE);
#endif
    while(running){
#ifdef _WIN32
        cv::waitKey(1);
#else
        QThread::msleep(1);
#endif
        while (inferSvc.in->size() < 3 && !inputs.empty()){
            lock.lock();
            auto m(std::move(inputs.front()));
            inputs.pop();
            lock.unlock();

            tmp.emplace(m);
            auto data = inferSvc.onnx -> read(tmp.back());
            inferSvc.lock->lock();
            inferSvc.in->emplace(data);
            inferSvc.lock->unlock();

            qDebug("[%lld] Send data to infer service", QThread::currentThreadId());
        }

        while (!inferSvc.out->empty() && !tmp.empty()){
            inferSvc.outlock->lock();
            auto result(std::move(inferSvc.out->front()));
            inferSvc.out->pop();
            inferSvc.outlock->unlock();
            qDebug("[%lld] Recv data to infer service", QThread::currentThreadId());

            auto outImage = inferSvc.onnx->postProcess(tmp.front(), result); 
            cv::putText(outImage, std::string(fps.fps(now())), {30, 30},
                cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 255));
#ifdef _WIN32
            cv::imshow(WINNAME, outImage);
            tmp.pop();
            cv::waitKey(1);
#else
            emit sImg(outImage);
            tmp.pop();
#endif
            // QImage Qtemp = QImage(outImage.data, outImage.cols, outImage.rows, outImage.step, QImage::Format_RGB888);
            // // emit sImg(Qtemp);
            // qDebug("[%lld] Send img to view", QThread::currentThreadId());
        }
    }
#ifdef _WIN32
    cv::destroyAllWindows();   
#endif
}

void VideoService::segMany(const QString &list)
{
    stopped = false;
    if (list.startsWith("DV:")){
        deviceid = list.split(":").rbegin() -> toInt();
        std::cout << "seg with video device: " << deviceid << std::endl;
        loop();
        return;
    }
    if (!list.endsWith(".txt")) {
        std::cout << "seg single image: " << qPrintable(list) << std::endl;
        auto mat0 = cv::imread(list.toStdString());
        if (mat0.empty()) { 
            std::cout << "fail to read: " << qPrintable(list) << std::endl;
            return;
        }
        inputSvc->lock.lock();
        inputSvc->inputs.emplace(std::move(mat0));
        inputSvc->lock.unlock();
        return;
    }
    QFile f(list);
    if(!f.open(QIODevice::ReadOnly | QIODevice::Text))
    {
        std::cout << "fail to open " << qPrintable(list) <<  std::endl;
        return;
    }
    QString str;
    while (!f.atEnd()){
        if (stopped) break;
        str = f.readLine();
        str = str.remove("\n");
        str = str.remove("\r");
        auto mat = cv::imread(str.toStdString());
        if (mat.empty()) {
            std::cout << "fail to read: " << qPrintable(str) << std::endl;
            continue;
        }



        while (inputSvc -> inputs.size() > 3){
            QThread::msleep(1);
            if (stopped) break;
        }
        inputSvc->lock.lock();
        inputSvc->inputs.push(mat);
        inputSvc->lock.unlock();
    }
    f.close();
    std::cout << "Video finished." << std::endl;
}

void VideoService::loop()
{
    // read from video
    stopped = false;
    cv::VideoCapture cap(deviceid);
    cv::Mat mat;
    while (cap.isOpened() && !stopped){
        cap >> mat;
        if (mat.empty()) continue;
        while (inputSvc -> inputs.size() > 3){
            QThread::msleep(1);
            if (stopped) break;
        }
        inputSvc->lock.lock();
        inputSvc->inputs.push(mat);
        inputSvc->lock.unlock();
    }
    qDebug("Video finished.");
}

OnnxService::OnnxService(QObject *parent):Daemon(parent)
{
    onnx = new MyONNX;
    lock = new QMutex;
    outlock = new QMutex;
    in = new std::queue<MyTensor>;
    out = new std::queue<std::vector<Ort::Value>>;
}

OnnxService::~OnnxService()
{
    if (onnx) {delete onnx; onnx = nullptr;}
    if (lock) {delete lock; lock = nullptr;}
    if (outlock) {delete outlock; outlock = nullptr;}
    if (in) {delete in; in = nullptr;}
    if (out) {delete out; out = nullptr;}
}


void OnnxService::loop()
{
    qDebug("Infer service: %lld", QThread::currentThreadId());
    MyTensor data;
    while (running){
        while (in->empty() && running) QThread::msleep(1);
        if (!running) break;
        lock->lock();
        data = in->front();
        lock->unlock();

        auto tmp(std::move(onnx -> one(data)));
        outlock -> lock();
        out -> emplace(std::move(tmp));
        outlock -> unlock();


        if (data.ptr) {delete[] data.ptr; data.ptr = nullptr;}
        lock->lock();
        in->pop();
        lock->unlock(); // pop input when finished.

        qDebug("[%lld] Infered.", QThread::currentThreadId());
    }
    qDebug("[%lld](infer) Quit.", QThread::currentThreadId());
}

Daemon::Daemon(QObject *parent): QObject(parent)
{
    this -> moveToThread(&me);
    QObject::connect(this, &Daemon::start,
                     this, &Daemon::slotStart);
    me.start();
}

Daemon::~Daemon()
{
    if (me.isRunning()) {
        running = false;
        me.quit();
        me.wait();
    }
}

void Daemon::stop()
{
    running = false;
}

void Daemon::slotStart()
{  
    running = true;
    loop();
    qDebug("[%lld] loop finished.", QThread::currentThreadId());
}

VideoService::VideoService(QObject *parent): Daemon(parent)
{
}

const char * FPS::fps(decltype(now()) t)
{
    times.push(t);
    if (times.size() < 2) return "";
    if (times.size() > 20) times.pop();
    sprintf(ss, "%8.2f FPS", times.size() * 1000.0 / (times.back() - times.front()));
    return ss;
}
