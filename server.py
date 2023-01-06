from flask import Flask, Response, render_template, jsonify, request, send_file
from camera import VideoCamera
import os
import time
import base64
import subprocess

from model import HubertModel
import config as cfg

wd = os.getcwd()

class FlaskApp(Flask):
    def __init__(self, *args, **kwargs):
        super(FlaskApp, self).__init__(*args, **kwargs)
        if not cfg.DEBUG:
            self.landmarks = {}
            self.model = HubertModel(wd, cfg.MODEL_PATH, cfg.PREDICTOR_PATH, cfg.MEAN_FACE_PATH)

app = FlaskApp(__name__)

video_camera = None
global_frame = None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/record_status', methods=['POST'])
def record_status():
    global video_camera 
    if video_camera == None:
        video_camera = VideoCamera()

    json = request.get_json()

    status = json['status']

    if status == "true":
        video_camera.start_record()
        return jsonify(result="started")
    else:
        video_camera.stop_record()
        return jsonify(result="stopped")

def video_stream():
    global video_camera 
    global global_frame

    if video_camera == None:
        video_camera = VideoCamera()
        
    while True:
        frame = video_camera.get_frame()

        if frame != None:
            global_frame = frame
            yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
        else:
            yield (b'--frame\r\n'
                            b'Content-Type: image/jpeg\r\n\r\n' + global_frame + b'\r\n\r\n')

@app.route('/video_viewer')
def video_viewer():
    return Response(video_stream(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/upload', methods=['POST'])
def upload_file():
    f = request.files['file']
    if f.filename == '':
        return
    # name = time.strftime("%Y%m%d-%H%M%S")

    # Retrieve unique name from server
    name = request.form['batchName']
    full_name = f'{name}{os.path.splitext(f.filename)[-1]}'
    f.save(f'{wd}/data/video-orig/{full_name}')

    # Get landmarks
    landmarks = [app.landmarks[i] for i in range(len(app.landmarks))]

    if cfg.DEBUG:
        prediction = "sample text"
    else:
        # prediction = app.model.transcribe(unique_name)
        try:
            prediction = app.model.get_transcription(name, full_name, landmarks)
        except Exception as e:
            print(e)
            prediction = "An error has occurred."

    response = Response(prediction)
    response.headers['filename'] = name + '.txt'

    app.landmarks = {}

    return response

@app.route('/upload_image', methods=['POST'])
def save_photo():
    # get the image data from the request
    image_data = request.json['imageString']
    image_num = request.json['imageNum']
    batch_name = request.json['batchName']

    # create dictionary for batch if needed
    # if batch_name not in app.landmarks:
        # app.landmarks[batch_name] = {}
    
    # decode base 64 image
    image_data_decoded = base64.b64decode(image_data)

    # save the image data to a folder based on batch name
    path = f'{wd}/data/temp/{batch_name}'
    if not os.path.exists(path):
        os.makedirs(path)
    with open(f'{path}/{image_num}.jpg', 'wb') as f:
        f.write(image_data_decoded)

    # TODO: GENERATE UNIQUE NAME ON SERVER SIDE FROM START OF BUTTON RECORDING
    # process image (don't need to save?)
    # TODO: spawn worker to do this concurrently?
    coords = app.model.get_landmark(image_data_decoded)
    app.landmarks[image_num] = coords
    # app.landmarks[batch_name][image_num] = coords

    # print(app.landmarks)

    return "OK"

if __name__=='__main__':
    if cfg.DEBUG:
        app.run(host='127.0.0.1', port=8080, threaded=True, debug=True)
    else:
        app.run(host='127.0.0.1', port=8080, threaded=True)