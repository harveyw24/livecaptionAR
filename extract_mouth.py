import dlib
import cv2
import tempfile
import subprocess
from multiprocessing import Pool

face_detector = dlib.get_frontal_face_detector()

def shape_to_list(shape):
	coords = []
	for i in range(0, 68):
		coords.append((shape.part(i).x, shape.part(i).y))
	return coords

def process_video(input_video_path, output_video_path, face_predictor_path):
    landmark_detector = dlib.shape_predictor(face_predictor_path)

    global detect_landmark

    # Obtain face landmark information
    def detect_landmark(image):
        face_rects = face_detector(image,1)             # Detect face
        if len(face_rects) < 1:                 # No face detected
            print("No face detected")
            return
        if len(face_rects) > 1:                  # Too many face detected
            print("Too many faces")
            return
        rect = face_rects[0]                    # Proper number of face
        landmark = landmark_detector(image, rect)   # Detect face landmarks
        landmark = shape_to_list(landmark)
        # landmark_buffer.append(landmark)
        return landmark
    
    LIP_MARGIN = 0.3                # Marginal rate for lip-only image.
    RESIZE = (96,96)                # Final image size

    vid = cv2.VideoCapture(input_video_path)       # Read video

    # Parse into frames
    frame_buffer = []               # A list to hold frame images
    frame_buffer_color = []         # A list to hold original frame images
    while(True):
        success, frame = vid.read()                # Read frame
        if not success:
            break                           # Break if no frame to read left
        gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)   # Convert image into grayscale
        frame_buffer.append(gray)                  # Add image to the frame buffer
        frame_buffer_color.append(frame)
    vid.release()

    with Pool(processes = 10) as pool:
        landmark_buffer = pool.map(detect_landmark, frame_buffer)
    
    print("done")

    # Crop images
    cropped_buffer = []
    for (i,landmark) in enumerate(landmark_buffer):
        lip_landmark = landmark[48:68]                                          # Landmark corresponding to lip
        lip_x = sorted(lip_landmark,key = lambda pointx: pointx[0])             # Lip landmark sorted for determining lip region
        lip_y = sorted(lip_landmark, key = lambda pointy: pointy[1])
        x_add = int((-lip_x[0][0]+lip_x[-1][0])*LIP_MARGIN)                     # Determine Margins for lip-only image
        y_add = int((-lip_y[0][1]+lip_y[-1][1])*LIP_MARGIN)
        crop_pos = (lip_x[0][0]-x_add, lip_x[-1][0]+x_add, lip_y[0][1]-y_add, lip_y[-1][1]+y_add)   # Crop image
        cropped = frame_buffer_color[i][crop_pos[2]:crop_pos[3],crop_pos[0]:crop_pos[1]]
        cropped = cv2.resize(cropped,(RESIZE[0],RESIZE[1]),interpolation=cv2.INTER_CUBIC)        # Resize
        cropped_buffer.append(cropped)

    # Save result to temp folder
    with tempfile.TemporaryDirectory() as tempdir:
        for (i,image) in enumerate(cropped_buffer):
            cv2.imwrite(str(tempdir) + "%d"%(i+1) + ".jpg", image)     # Write lip image

        # Save video
        command = f'ffmpeg -i {str(tempdir)}%d.jpg {output_video_path}'
        subprocess.call(command, shell=True)


# import dlib
# import cv2
# import os
# import tempfile
# import subprocess
# from functools import partial
# from multiprocessing import Pool

# face_detector = dlib.get_frontal_face_detector()

# def shape_to_list(shape):
# 	coords = []
# 	for i in range(0, 68):
# 		coords.append((shape.part(i).x, shape.part(i).y))
# 	return coords

# # Obtain face landmark information
# def detect_landmark(image, landmark_detector):
#     face_rects = face_detector(image,1)             # Detect face
#     if len(face_rects) < 1:                 # No face detected
#         print("No face detected")
#         return
#     if len(face_rects) > 1:                  # Too many face detected
#         print("Too many faces")
#         return
#     rect = face_rects[0]                    # Proper number of face
#     landmark = landmark_detector(image, rect)   # Detect face landmarks
#     landmark = shape_to_list(landmark)
#     # landmark_buffer.append(landmark)
#     return landmark

# def process_video(input_video_path, output_video_path, face_predictor_path, mean_face_path):
#     landmark_detector = dlib.shape_predictor(face_predictor_path)
    
#     TEMP_PATH = './process_video/'       # The path that the result images will be saved
#     LIP_MARGIN = 0.3                # Marginal rate for lip-only image.
#     RESIZE = (96,96)                # Final image size

#     vid = cv2.VideoCapture(input_video_path)       # Read video

#     # Parse into frames
#     frame_buffer = []               # A list to hold frame images
#     frame_buffer_color = []         # A list to hold original frame images
#     while(True):
#         success, frame = vid.read()                # Read frame
#         if not success:
#             break                           # Break if no frame to read left
#         gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)   # Convert image into grayscale
#         frame_buffer.append(gray)                  # Add image to the frame buffer
#         frame_buffer_color.append(frame)
#     vid.release()

#     with Pool(processes = 10) as pool:
#         landmark_buffer = pool.map(partial(detect_landmark, landmark_detector=landmark_detector), frame_buffer)
    
#     print("done")

#     # Crop images
#     cropped_buffer = []
#     for (i,landmark) in enumerate(landmark_buffer):
#         lip_landmark = landmark[48:68]                                          # Landmark corresponding to lip
#         lip_x = sorted(lip_landmark,key = lambda pointx: pointx[0])             # Lip landmark sorted for determining lip region
#         lip_y = sorted(lip_landmark, key = lambda pointy: pointy[1])
#         x_add = int((-lip_x[0][0]+lip_x[-1][0])*LIP_MARGIN)                     # Determine Margins for lip-only image
#         y_add = int((-lip_y[0][1]+lip_y[-1][1])*LIP_MARGIN)
#         crop_pos = (lip_x[0][0]-x_add, lip_x[-1][0]+x_add, lip_y[0][1]-y_add, lip_y[-1][1]+y_add)   # Crop image
#         cropped = frame_buffer_color[i][crop_pos[2]:crop_pos[3],crop_pos[0]:crop_pos[1]]
#         cropped = cv2.resize(cropped,(RESIZE[0],RESIZE[1]),interpolation=cv2.INTER_CUBIC)        # Resize
#         cropped_buffer.append(cropped)

#     # Save result to temp folder
#     tempdir = tempfile.TemporaryDirectory()
#     for (i,image) in enumerate(cropped_buffer):
#         cv2.imwrite(str(tempdir) + "%d"%(i+1) + ".jpg", image)     # Write lip image

#     # Save video
#     command = f'ffmpeg -i {str(tempdir)}%d.jpg {output_video_path}'
#     subprocess.call(command, shell=True)

#     os.removedirs(str(tempdir))