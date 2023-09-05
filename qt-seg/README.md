# Intro

- 代码在以下三个平台测试ok（onnxruntime直接官网下）

- 用的是CMake+Ninja

- Windows 22H2 (19045.3208)

  - MSVC2017 64bit (14.16.27023)

  - Qt5 (5.14.2)

  - Win10SDK (10.0.20348.0)

- MacOS Ventura (13.4.1)

  - clang 14.0.3

  - Qt6 (6.5.1)

  - opencv 4.8.0 (brew)

- Ubuntu 18.04 (容器，非物理机)

  - gcc 8.2.0 (编译器)

  - Qt6 (6.5.1)

  - libopencv-???-dev (需要的包见CMakeLists.txt)

  - 说明：宿主机glibc低于2.28，需要单独编译gcc8.2.0+glibc2.28，
    并额外修改安装的Qt的interpreter（patchelf），
    需要patch的程序主要是rcc和moc


# Usage

- 启动后，先修改onnx模型路径，点击加载

- 然后修改txt或者视频设备路径，然后开始分割

- 点击停止，停止状态下才可以修改配置或者模型（防止bug）

- 退出程序

# TODO

- 内存泄漏检测

- 更统一的CMakeLists

- 更完善的后处理
  （目前是手写的nms和mat转tensor，考虑调包）

