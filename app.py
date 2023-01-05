import streamlit as st
from streamlit_webrtc import WebRtcMode, webrtc_streamer
import av
import cv2
import os
import threading
from collections import deque
from typing import List
import numpy as np
import pydub
import time
import queue
from aiortc.contrib.media import MediaRecorder
import dlib
import tempfile
import subprocess

from model import HubertModel
from avhubert import extract_roi
from avhubert.preparation.align_mouth import landmarks_interpolate, crop_patch, write_video_ffmpeg

WD = os.getcwd()
lock = threading.Lock()
img_container = {"img": None}
audio_container = {"audio": None}

video = []
landmarks = []
audio = pydub.AudioSegment.empty()

def save_audio():
    audio.export("output_audio.wav", format="wav", bitrate='16k', parameters=["-ac", "1", "-ar", "16000"])

def save_video():
    print(len(video))
    print(video[0])
    preprocessed_landmarks = landmarks_interpolate(landmarks)
    mean_face_path = f"{WD}/data/misc/20words_mean_face.npy"
    mean_face_landmarks = np.load(mean_face_path)

    stablePntsIDs = [33, 36, 39, 42, 45]
    STD_SIZE = (256, 256)

    frameSize = (636, 360)
    fps = 25.0
    out = cv2.VideoWriter(WD + 'output_vid.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, frameSize)
    result_rgb = cv2.cvtColor(video[0], cv2.COLOR_BGR2RGBA)
    cv2.imwrite("output_sample_frame.jpg", video[-1])
    print(np.shape(video[0]))
    print(len(video))
    print(len(landmarks))
    for (i,image) in enumerate(video):
            cv2.imwrite('temp_pics/' + "%d"%(i+1) + ".jpg", image)     # Write lip image

    with tempfile.TemporaryDirectory() as tempdir:
        for (i,image) in enumerate(video):
            cv2.imwrite(str(tempdir) + "%d"%(2*i+1) + ".jpg", image)     # Write lip image
            cv2.imwrite(str(tempdir) + "%d"%(2*i+2) + ".jpg", image)     # Write lip image

        # Save video
        # command = f'ffmpeg -r 15 -i {str(tempdir)}%d.jpg -vf minterpolate=fps=25:mi_mode=dup output_vid_2.mp4'
        command = f'ffmpeg -r 25 -i {str(tempdir)}%d.jpg output_vid_2.mp4'
        subprocess.call(command, shell=True)


    for i in range(len(video)):
        out.write(video[i])
    out.release()
    print("produced video")

    input_vid_path = WD + '/output_vid_2.mp4'
    print(input_vid_path)
    rois = crop_patch(input_vid_path, preprocessed_landmarks, mean_face_landmarks, stablePntsIDs, STD_SIZE, 
                        window_margin=12, start_idx=48, stop_idx=68, crop_height=96, crop_width=96)
    print("cropped roi")
    print(rois)
    mouth_roi_path = WD + '/output_cropped.mp4'
    print(mouth_roi_path)
    write_video_ffmpeg(rois, mouth_roi_path, "ffmpeg")
    print("done")


face_predictor_path = f"{WD}/data/misc/shape_predictor_68_face_landmarks.dat"
if 'detector' not in st.session_state:
    st.session_state.detector = dlib.get_frontal_face_detector()

if 'predictor' not in st.session_state:
    st.session_state.predictor = dlib.shape_predictor(face_predictor_path)

def find_landmark(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    rects = st.session_state.detector(gray, 1)
    coords = None
    for (_, rect) in enumerate(rects):
        shape = st.session_state.predictor(gray, rect)
        coords = np.zeros((68, 2), dtype=np.int32)
        for i in range(0, 68):
            coords[i] = (shape.part(i).x, shape.part(i).y)
    return coords
    
def video_frame_callback(frame):
    img = frame.to_ndarray(format="bgr24")
    with lock:
        img_container["img"] = img
        video.append(img)
    return frame

def main():
    st.header("Real Time Speech-to-Text")

    # if 'model' not in st.session_state:
        # st.session_state.model = HubertModel(f"{WD}/data/misc/large_noise_pt_noise_ft_433h.pt", WD)
        # print("Model finished loading.")

    run_app()

def run_app():

    frames_deque: deque = deque([])
    video_deque: deque = deque([])

    async def queued_audio_frames_callback(
        frames: List[av.AudioFrame],
    ) -> av.AudioFrame:
        with lock:
            frames_deque.extend(frames)

        # Return empty frames to be silent.
        new_frames = []
        for frame in frames:
            input_array = frame.to_ndarray()
            new_frame = av.AudioFrame.from_ndarray(
                np.zeros(input_array.shape, dtype=input_array.dtype),
                layout=frame.layout.name,
            )
            new_frame.sample_rate = frame.sample_rate
            new_frames.append(new_frame)

        return new_frames

    async def queued_video_frames_callback(
        frames: List[av.VideoFrame],
    ) -> av.VideoFrame:
        with lock:
            video_deque.extend(frames)
        return frames

    webrtc = webrtc_streamer(
        key="camera", 
        mode=WebRtcMode.SENDRECV,
        # video_frame_callback=video_frame_callback,
        queued_video_frames_callback=queued_video_frames_callback,
        queued_audio_frames_callback=queued_audio_frames_callback,
        audio_receiver_size=256,
        media_stream_constraints={
            "audio": {
                "channelCount": 1,
                "sampleRate": 16000,
                "volume": 1,
                "echoCancellation": False,
                "noiseSuppression": True
            },
            "video": {
                "width": {"max": 640},
                "height": {"max": 480},
                "frameRate": {"exact": 12.5}
            }
        },
        on_video_ended=save_video,
        on_audio_ended=save_audio
    )

    status_indicator = st.empty()

    if not webrtc.state.playing:
        return
    
    status_indicator.write("Loading...")
    text_output = st.empty()


    async def processLandmarks(video_frames):
        for img in video_frames:
            res = find_landmark(img)
            landmarks.append(res)
            landmarks.append(res)


    while True:
        if webrtc.state.playing:
            # Grab video
            # with lock:
            #     img = img_container["img"]

            # if img is None:
            #     continue

            # video.append(img)
            video_frames = []
            with lock:
                while len(video_deque) > 0:
                    frame = video_deque.popleft()
                    video_frames.append(frame.to_ndarray(format="bgr24"))
            
            processLandmarks(video_frames)
            video.extend(video_frames)

            # Grab audio
            audio_frames = []
            sound_chunk = pydub.AudioSegment.empty()

            with lock:
                while len(frames_deque) > 0:
                    frame = frames_deque.popleft()
                    audio_frames.append(frame)

            if len(audio_frames) == 0:
                time.sleep(0.1)
                status_indicator.write("No frame arrived.")
                continue

            status_indicator.write("Running. Say something!")

            for audio_frame in audio_frames:
                sound = pydub.AudioSegment(
                    data=audio_frame.to_ndarray().tobytes(),
                    sample_width=audio_frame.format.bytes,
                    frame_rate=audio_frame.sample_rate,
                    channels=len(audio_frame.layout.channels),
                )
                sound_chunk += sound

            global audio

            if len(sound_chunk) > 0:
                sound_chunk = sound_chunk.set_channels(1).set_frame_rate(24000)
                print(type(sound_chunk))
                audio += sound_chunk
                buffer = np.array(sound_chunk.get_array_of_samples())
                # text = stream.intermediateDecode()
                # text_output.markdown(f"**Text:** {text}")
        else:
            status_indicator.write("Stopped.")
            break




if __name__ == "__main__":
    import os

    DEBUG = os.environ.get("DEBUG", "false").lower() not in ["false", "no", "0"]

    main()