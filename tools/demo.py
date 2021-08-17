from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals


from photodrive.ytcrol import *
from photodrive.hikrobot import *
import sys
sys.path.append(r"D:\MVS\Development\Samples\Python\MvImport")
from MvCameraControl_class import *

import os
import argparse

import cv2
import torch
import serial
import numpy as np
from glob import glob

from pysot.core.config import cfg
from pysot.models.model_builder import ModelBuilder
from pysot.tracker.tracker_builder import build_tracker

torch.set_num_threads(1)

parser = argparse.ArgumentParser(description='tracking demo')
parser.add_argument('--config', type=str, help='config file')
parser.add_argument('--snapshot',  type=str, help='model name')
parser.add_argument('--video_name', default='', type=str,
                    help='videos or image files')
parser.add_argument('--camara', type=str,default='net', help='hk or net')
parser.add_argument('--control', type=str,default='nocontrol', help='nocontrol or noauto or auto')
parser.add_argument('--record', type=str,default='no', help='yes or no')
args = parser.parse_args()


def get_frames(video_name):

    if not video_name:
        if args.camara == "hk":
            ##########################海康相机驱动##############################
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

            nDataSize = 2448 * 2048 #分辨率，这里不能更改，只能写设备分辨率，更改的话去帧之后用resize
            pData = (c_ubyte * nDataSize)()
            stFrameInfo = MV_FRAME_OUT_INFO_EX()
            memset(byref(stFrameInfo), 0, sizeof(stFrameInfo))
            #################################################################
        elif args.camara == "net":
            ############################网络相机################################
            cap = cv2.VideoCapture(0)
            #################################################################
        else:
            print("please set whitch camara")
            assert (0)
        # warmup
        for i in range(1):
            cv2.waitKey(1000)
        while True:
            if args.camara == "hk":
                ret = cam.MV_CC_GetOneFrameTimeout(pData, nDataSize, stFrameInfo, 1000)
                ret = 1 - ret   #海康相机0表示读到视频，1表示没读到，和正常思路不符，因此取反
                frame = np.asarray(pData).reshape((stFrameInfo.nHeight, stFrameInfo.nWidth))
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            elif args.camara == "net":
                ret, frame = cap.read()
            frame = cv2.resize(frame, (1280, 1024))  # 改分辨率
            if ret: #1表示读到视频
                yield frame
            else:
                break
    elif video_name.endswith('avi') or \
        video_name.endswith('mp4'):
        cap = cv2.VideoCapture(args.video_name)
        while True:
            ret, frame = cap.read()
            if ret:
                yield frame
            else:
                break
    else:
        images = glob(os.path.join(video_name, '*.jp*'))
        images = sorted(images,
                        key=lambda x: int(x.split('/')[-1].split('.')[0]))
        for img in images:
            frame = cv2.imread(img)
            yield frame


def main():
    # load config
    cfg.merge_from_file(args.config)
    cfg.CUDA = torch.cuda.is_available() and cfg.CUDA
    device = torch.device('cuda' if cfg.CUDA else 'cpu')

    #box缓存
    box_temp=[0, 0, 0, 0]

    # create model
    model = ModelBuilder()

    # load model
    model.load_state_dict(torch.load(args.snapshot,
        map_location=lambda storage, loc: storage.cpu()))
    model.eval().to(device)

    # build tracker
    tracker = build_tracker(model)

    #海康相机串口初始化
    if args.camara == "noauto" :
        ss = ser('COM3', 19200, 0.01)
        ss.openEngine

    #是否录制视频
    if args.record == "yes":
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(r'C:\Users\dell\Videos\output.avi', fourcc, 20.0, (1280, 1024),True)

    first_frame = True

    if args.video_name:
        video_name = args.video_name.split('/')[-1].split('.')[0]
    else:
        video_name = 'webcam'
    cv2.namedWindow(video_name, cv2.WND_PROP_FULLSCREEN)
    for frame in get_frames(args.video_name):
        if first_frame:
            cv2.putText(frame, "Enter ROI", (100, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                        (0, 255, 0), 2)
            try:
                init_rect = cv2.selectROI(video_name, frame, False, False)
            except:
                exit()
            tracker.init(frame, init_rect)
            first_frame = False
        else:
            outputs = tracker.track(frame)

            if 'polygon' in outputs:
                polygon = np.array(outputs['polygon']).astype(np.int32)
                cv2.polylines(frame, [polygon.reshape((-1, 1, 2))],
                              True, (0, 255, 0), 3)
                mask = ((outputs['mask'] > cfg.TRACK.MASK_THERSHOLD) * 255)
                mask = mask.astype(np.uint8)
                mask = np.stack([mask, mask*255, mask]).transpose(1, 2, 0)
                frame = cv2.addWeighted(frame, 0.77, mask, 0.23, -1)
            else:
                if outputs['lost'] == 0:
                    bbox = list(map(int, outputs['bbox']))
                    cv2.putText(frame, "Tracking OK", (100, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                                (0, 255, 0), 2)
                    cv2.rectangle(frame, (bbox[0], bbox[1]),
                                 (bbox[0]+bbox[2], bbox[1]+bbox[3]),
                                 (0, 255, 0), 3)
                    if args.control == "auto":
                    ###################根据识别结果自动控制云台#####################
                    #本帧与下一帧偏差小于5个像素才控制云台，bias为帧偏移量记录数据
                        box_bias=abs(bbox[0] - box_temp[0])+ \
                                 abs(bbox[1] - box_temp[1])+ \
                                 abs(bbox[2] - box_temp[2])+ \
                                 abs(bbox[3] - box_temp[3])
                        for i in range(4):
                            box_temp[i] = bbox[i]     #帧缓存
                        if bbox[2]>0 and bbox[3]>0 and box_bias>20:
                            ss.control(bbox[0],bbox[1],bbox[2],bbox[3],1280,1024) #帧偏移量大则控制云台移动
                        else:
                            ss.stop()
                    #############################################################
                else:
                    cv2.putText(frame, "Tracking failure detected", (100, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                              (0, 0, 255), 2)

                ##########程序原函数###############
                # bbox = list(map(int, outputs['bbox']))
                # cv2.rectangle(frame, (bbox[0], bbox[1]),
                #              (bbox[0]+bbox[2], bbox[1]+bbox[3]),
                #              (0, 255, 0), 3)
                ###################################
            #print(outputs['best_score'])
            cv2.imshow(video_name, frame)


            first_frame_temp = cv2.waitKeyEx(10)    #2490368 up 2621440 down 2424832 left  2555904 right
            if args.control == "noauto" :
                #################################通过方向键手动控制云台#########################
                if first_frame_temp == 2490368:
                    ss.up()
                    time.sleep(0.15)
                    ss.stop()
                    print("now go up,",first_frame_temp)
                elif first_frame_temp == 2621440:
                    ss.down()
                    time.sleep(0.15)
                    ss.stop()
                    print("now go down,", first_frame_temp)
                elif first_frame_temp == 2424832:
                    ss.left()
                    time.sleep(0.15)
                    ss.stop()
                    print("now go left,", first_frame_temp)
                elif first_frame_temp == 2555904:
                    ss.right()
                    time.sleep(0.15)
                    ss.stop()
                    print("now go right,", first_frame_temp)
                elif first_frame_temp != -1:
                    first_frame = True
                else:
                    pass
                ##############################################################################
            if args.control == "nocontrol":
                lt_temp = first_frame_temp
                ###########重新选择ROI##############
                if lt_temp != -1:
                    first_frame = True
                lt_temp = -1
                #############################
            #录制视频
            if args.record == "yes":
                out.write(frame)



if __name__ == '__main__':
    main()
