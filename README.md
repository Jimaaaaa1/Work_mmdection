1. deepsort 目录下是deepsort代码
2. qt-seg 目录下是Qt+Yolov5-seg代码
3. trackball 目录是小球数据集
4. main.avi 是路口视频

5. opencv-install 是安装的opencv位置【可以移动，相应修改CMAKE配置即可】
6. gitrepos 是写代码时用于同步win和nano的git【可以删除】
7. libclog 是编译opencv时用到的【应该可以删除】
8. Qt5.14.2-base-quick-ok.tar 是本机编译的qt，包含qtbase、qtdeclaritive、qtquickcontrols、qtquickcontrols
9. Qt5.14.2.tar 是跨平台编译的qt，包含所有模块，但是不能通过X11显示（不含xcb），同时需要把上面的qmake、uic、moc、rcc复制进去使用