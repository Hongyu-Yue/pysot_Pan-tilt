A Pan-tilthk control Program with Haikang camera.Based on pysot.
Bolg：
https://blog.csdn.net/weixin_56184890/article/details/119733363?spm=1001.2014.3001.5501

修改了demo.py，增加了photodrive文件夹，内有两个文件，分别为云台和相机控制。需要有相应的硬件才能运行，主要需要云台和海康的长焦相机。

目前使用的相机是海康的一款长焦相机，不能直接通过cv2调用，需要调用驱动，程序里也增加了相机调用的模块。
这部分代码位置：photodrive/hikrobot.py

以及程序里增加了云台相机的控制代码，主要是通过候选框的位置实时控制云台转动，让相机能够实时跟踪。
这部分代码位置：photodrive/ytcrol.py

其他改动主要针对两个问题：
1.如果跟丢不会报告，而是ROI定在跟丢前地址不动。
2.一旦选择了ROI之后，无法重新选择。

