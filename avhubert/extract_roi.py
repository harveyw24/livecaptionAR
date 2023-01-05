'''
Author: Harvey
Date: 2022-11-23 16:06:03
LastEditors: Harvey
LastEditTime: 2022-11-23 23:39:54
Description: 
'''
import dlib, cv2, os
import numpy as np
import skvideo
import skvideo.io
from tqdm import tqdm
from multiprocessing import Pool
from functools import partial
from .preparation.align_mouth import landmarks_interpolate, crop_patch, write_video_ffmpeg
# from IPython.display import HTML
from base64 import b64encode

def play_video(video_path, width=200):
  # mp4 = open(video_path,'rb').read()
  # data_url = "data:video/mp4;base64," + b64encode(mp4).decode()
  # return HTML(f"""
  # <video width={width} controls>
  #       <source src="{data_url}" type="video/mp4">
  # </video>
  # """)
  print("Playing video...")
  cap = cv2.VideoCapture(video_path)
  while(cap.isOpened()):
      
  # Capture frame-by-frame
      ret, frame = cap.read()
      if ret == True:
      # Display the resulting frame
          cv2.imshow('Frame', frame)
          
      # Press Q on keyboard to exit
          if cv2.waitKey(25) & 0xFF == ord('q'):
              break
  
  # Break the loop
      else:
          break

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


def preprocess_video(input_video_path, output_video_path, face_predictor_path, mean_face_path):
  detector = dlib.get_frontal_face_detector()
  predictor = dlib.shape_predictor(face_predictor_path)
  STD_SIZE = (256, 256)
  mean_face_landmarks = np.load(mean_face_path)
  stablePntsIDs = [33, 36, 39, 42, 45]
  videogen = skvideo.io.vread(input_video_path)
  frames = np.array([frame for frame in videogen])
  landmarks = []

  global detect_landmark

  def detect_landmark(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    rects = detector(gray, 1)
    coords = None
    for (_, rect) in enumerate(rects):
        shape = predictor(gray, rect)
        coords = np.zeros((68, 2), dtype=np.int32)
        for i in range(0, 68):
            coords[i] = (shape.part(i).x, shape.part(i).y)
    return coords

  with Pool(processes = 8) as pool:
    landmarks = pool.map(detect_landmark, frames)

  # for frame in tqdm(frames):
  #   landmark = detect_landmark(frame, detector, predictor)
  #   landmarks.append(landmark)

  preprocessed_landmarks = landmarks_interpolate(landmarks)
  rois = crop_patch(input_video_path, preprocessed_landmarks, mean_face_landmarks, stablePntsIDs, STD_SIZE, 
                        window_margin=12, start_idx=48, stop_idx=68, crop_height=96, crop_width=96)
  write_video_ffmpeg(rois, output_video_path, "ffmpeg")
  return


def test_create_file(target_path):
    import os,pickle,shutil,tempfile
    os.makedirs(target_path, exist_ok=True)
    f = open(f"{target_path}/demofile.txt", "x")
    f.write("Now the file has more content!")
    f.close()

    if os.path.isfile(f"{target_path}/demofile.txt"):
        print("deleting file")
        os.remove(f"{target_path}/demofile.txt")

    # for i_roi, roi in enumerate(rois):
    #     cv2.imwrite(os.path.join(tmp_dir, str(i_roi).zfill(decimals)+'.png'), roi)
    # list_fn = os.path.join(tmp_dir, "list")
    # with open(list_fn, 'w') as fo:
    #     fo.write("file " + "'" + tmp_dir+'/%0'+str(decimals)+'d.png' + "'\n")
    # ## ffmpeg
    # if os.path.isfile(target_path):
    #     os.remove(target_path)
    # cmd = [ffmpeg, "-f", "concat", "-safe", "0", "-i", list_fn, "-q:v", "1", "-r", str(fps), '-y', '-crf', '20', target_path]
    # pipe = subprocess.run(cmd, stdout = subprocess.PIPE, stderr = subprocess.STDOUT)
    # # rm tmp dir
    # shutil.rmtree(tmp_dir)
    return