import sys
sys.path.append(r"D:\MVS\Development\Samples\Python\MvImport")
from MvCameraControl_class import *
import numpy as np
import cv2
def videocapture():

    deviceList = MV_CC_DEVICE_INFO_LIST()
    nRet = MvCamera.MV_CC_EnumDevices(MV_USB_DEVICE, deviceList)
    if nRet != 0:
        print("enum devices fail! nRet[0x%x]" % nRet)
        sys.exit()
    if deviceList.nDeviceNum == 0:
        print("find no device!")
        sys.exit()
    print("Find %d devices!" % deviceList.nDeviceNum)

    # 创建相机实例并创建句柄
    nConnectionNum = 0
    stDeviceList = cast(deviceList.pDeviceInfo[int(nConnectionNum)], POINTER(MV_CC_DEVICE_INFO)).contents
    cam = MvCamera()
    nRet = cam.MV_CC_CreateHandle(stDeviceList)
    if nRet != 0:
        print("create handle fail! nRetnRet[0x%x]" % nRet)
        sys.exit()

    # 打开设备
    nRet = cam.MV_CC_OpenDevice(MV_ACCESS_Exclusive, 0)
    if nRet != 0:
        print("open device fail! nRet[0x%x]" % nRet)
        sys.exit()

    # 开始取流
    nRet = cam.MV_CC_StartGrabbing()
    if nRet != 0:
        print("start grabbing fail! nRet[0x%x]" % nRet)
        sys.exit()

    nDataSize = 2448 * 2048
    pData = (c_ubyte * nDataSize)()
    stFrameInfo = MV_FRAME_OUT_INFO_EX()
    memset(byref(stFrameInfo), 0, sizeof(stFrameInfo))
    return pData, nDataSize, stFrameInfo
    

#     while True:
#         ret = cam.MV_CC_GetOneFrameTimeout(pData, nDataSize, stFrameInfo, 1000)
#         #print(ret)
#         frame = np.asarray(pData).reshape((stFrameInfo.nHeight, stFrameInfo.nWidth))
#         cv2.namedWindow("teswell", cv2.WINDOW_NORMAL)
#         cv2.imshow('teswell', frame) #显示画面
#         cv2.waitKey(1)
#
#     nRet = cam.MV_CC_StopGrabbing()
#     if nRet != 0:
#         print("stop grabbing fail! nRet[0x%x]" % nRet)
#         sys.exit()
#
#     nRet = cam.MV_CC_CloseDevice();
#     if nRet != 0:
#         print("close device fail! nRet[0x%x]" % nRet)
#         sys.exit()
#
#
#     cv2.destroyAllWindows() #释放所有显示图像窗口
#
# if __name__ == '__main__' :
#
#     videocapture()

