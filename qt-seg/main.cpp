#include <QGuiApplication>
#include <QQmlApplicationEngine>
#include "controller.h"
#include <iostream>
#include <QElapsedTimer>
#include <QQmlContext>
// yolov5 export: ["--weights", "last.pt", "--include", "onnx", "--imgsz", "288", "384"]
// yolov5 export: git checkout v7.0 && python export.py --weights last.pt --include onnx --imgsz 288 384
// yolov8 export: yolo export model=yolov8n-seg.pt format=onnx imgsz=288,384

int main(int argc, char *argv[])
{
    QCoreApplication::setAttribute(Qt::AA_EnableHighDpiScaling);

    QGuiApplication app(argc, argv);
    qRegisterMetaType<cv::Mat>("cv::Mat");

    QQmlApplicationEngine engine;
    Controller con;

//    engine.addImageProvider(QLatin1String("pImg"), &con.provider);
    engine.rootContext() -> setContextProperty("con", &con);
    const QUrl url(QStringLiteral("qrc:/main.qml"));
    QObject::connect(&engine, &QQmlApplicationEngine::objectCreated,
                     &app, [url](QObject *obj, const QUrl &objUrl) {
        if (!obj && url == objUrl){
            std::cout << "fail to read qml" << std::endl;
            QCoreApplication::exit(-1);
        }
    }, Qt::QueuedConnection);
    engine.load(url);
    return app.exec();
}
