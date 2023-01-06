'''
Author: Harvey
Date: 2022-11-23 16:06:03
LastEditors: Harvey
LastEditTime: 2022-11-23 23:39:54
Description: 
'''
import cv2
import numpy as np
from .preparation.align_mouth import landmarks_interpolate, crop_patch, write_video_ffmpeg

def find_landmark(image, detector, predictor):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    rects = detector(gray, 1)
    coords = None
    for (_, rect) in enumerate(rects):
        shape = predictor(gray, rect)
        coords = np.zeros((68, 2), dtype=np.int32)
        for i in range(0, 68):
            coords[i] = (shape.part(i).x, shape.part(i).y)
    return coords

def crop_video(input_video_path, output_video_path, landmarks, mean_face_landmarks):
    STD_SIZE = (256, 256)
    stablePntsIDs = [33, 36, 39, 42, 45]
    preprocessed_landmarks = landmarks_interpolate(landmarks)
    rois = crop_patch(input_video_path, preprocessed_landmarks, mean_face_landmarks, stablePntsIDs, STD_SIZE, 
                        window_margin=12, start_idx=48, stop_idx=68, crop_height=96, crop_width=96)
    write_video_ffmpeg(rois, output_video_path, "ffmpeg")