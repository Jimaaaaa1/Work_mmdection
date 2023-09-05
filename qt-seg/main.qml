import QtQuick 2.12
import QtQuick.Window 2.12
import QtQuick.Controls 2.12

Window {
    visible: true
    width: 800
    height: 300
    title: qsTr("Hello World")

    onClosing:{
        con.quit();
        Qt.quit();
    }

    Rectangle{
        anchors.fill: parent
        anchors.margins: {
            top: 50
            left: 50
            right: 50
            bottom: 50
        }

        Column{
            spacing: 20

            Row{
                Label{
                    text: qsTr("为防止bug,修改配置时,请在停止状态下进行")
                }
            }

            Row{
                spacing: 20

                TextField{
                    width: 200
                    id: img
                    selectByMouse: true
                    text: "l.txt"
                }

                Button{
                    text: "分割"
                    width: 90
                    onClicked: {
                        con.seg(img.text)
                    }
                }

                Button{
                    text: "停止"
                    width: 90
                    onClicked: {
                        con.stop();
                    }
                }
            }

            Row {

                spacing: 20

                Label {
                    text: "conf_th:"
                }

                TextField{
                    width: 120
                    id: confth
                    selectByMouse: true
                    text: "0.25"
                }

                Label {
                    text: "iou_th:"
                }

                TextField{
                    width: 120
                    id: iouth
                    selectByMouse: true
                    text: "0.45"
                }

                Button{
                    text: "确认设置"
                    onClicked:{
                        
                    }
                }
                
            }

            Row {
                spacing: 20

                Label{
                    text: "model:"
                }

                TextField{
                    width: 200
                    id: modelname
                    selectByMouse: true
                    text: "best.onnx"
                }

                Button{
                    text: "加载模型"
                    onClicked: {
                        con.setModel(modelname.text)
                    }                    
                }

            }


            // Row{
            //     Image {
            //         cache: false
            //         width: 640
            //         height: 640
            //         id: image
            //     }
            //     Connections {
            //         target: con
            //         onCallQmlRefeshImg: {
            //             image.source = ""
            //             image.source = "image://pImg"
            //         }
            //     }
            // }
        }
    }
}
