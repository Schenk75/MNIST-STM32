# STM32F103指南者手写体识别

## 功能

- 使用OV7725摄像头拍摄，在液晶屏实时显示
- 通过串口将拍摄图像传至上位机
- 上位机使用pytorch模型进行分类

## 操作

- train.py训练模型，server.py打开服务进行
- 按下K1键拍照截图，等待LED灯变绿，拍摄并识别完成，并在液晶屏上显示分类结果
- 按下K2键返回拍摄界面，LED灯变蓝，可重新拍照
