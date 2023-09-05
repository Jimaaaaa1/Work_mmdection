#ifndef CONTROLLER_H
#define CONTROLLER_H
#include "myonnx.h"
#include <QObject>
#include <QThread>
// #include <QImage>
// #include <QQuickImageProvider>
// #include <QTcpSocket>
// #include <QTcpServer>
// #include <QBuffer>
#include <QDebug>
#include <QMutex>
#include <queue>


class Daemon: public QObject{
    Q_OBJECT
public:
    explicit Daemon(QObject* parent=nullptr);
    ~Daemon();
signals:
    void start();
public:
    Q_INVOKABLE void stop();

private slots:
    void slotStart();
protected:
    virtual void loop() = 0;
    QThread me;
    bool running = false;
};


// class ImageProvider : public QQuickImageProvider
// {
// public:
//     ImageProvider();
//     QImage requestImage(const QString &id, QSize *size, const QSize &requestedSize);
//     QPixmap requestPixmap(const QString &id, QSize *size, const QSize &requestedSize);
//     QImage img;
// };


/*
    onnx的推理线程
*/
class OnnxService: public Daemon
{
    Q_OBJECT
public:
    explicit OnnxService(QObject *parent = nullptr);
    ~OnnxService();
    MyONNX *onnx = nullptr;
    QMutex *lock, *outlock;
    std::queue<MyTensor> *in;
    std::queue<std::vector<Ort::Value>> *out;
    void loop();
};



/*
    Onnx的输入线程
*/
class OnnxInputService : public Daemon
{
    Q_OBJECT
public:
    explicit OnnxInputService(QObject *parent = nullptr);
    ~OnnxInputService();
    OnnxService* inferSvc;
    QMutex lock;
    std::queue<cv::Mat> inputs;

public slots:
    void loop();
signals:
    void sImg(const cv::Mat &); // 显示图片的信号，将信号发送到这里，就会刷新图片
};

class VideoService : public Daemon
{
    Q_OBJECT
public:
    explicit VideoService(QObject *parent = nullptr);
    OnnxInputService* inputSvc;
    bool stopped;
public slots:
    void segMany(const QString&); // 接收一个list作为输入
    void loop();

private:
    int deviceid = -1;
};


class Controller : public QObject
{
    Q_OBJECT
public:
    explicit Controller(QObject *parent = nullptr);
    ~Controller();
    Q_INVOKABLE void seg(const QString&);
    Q_INVOKABLE void stop();
    Q_INVOKABLE void quit();

    Q_INVOKABLE void setNMS(const QString&, const QString&);
    Q_INVOKABLE void setModel(const QString&);
public slots:
    void showImage(const cv::Mat &);
private:
    OnnxService inferSvc;
    OnnxInputService inputSvc;
    VideoService vidSvc;
signals:
    void sSeg(const QString&);
};


class FPS{
public:
    std::queue<decltype(now())> times;
    const char * fps(decltype(now()));
private:
    char ss[32];
};

#endif // CONTROLLER_H
