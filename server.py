from flask import Flask, Response, render_template, jsonify, request, send_file
from camera import VideoCamera
import os
import time
import base64
import subprocess

from model import HubertModel

wd = os.getcwd()

class FlaskApp(Flask):
    def __init__(self, *args, **kwargs):
        super(FlaskApp, self).__init__(*args, **kwargs)
        self.model = HubertModel(f"{wd}/data/misc/large_noise_pt_noise_ft_433h.pt", wd)

app = FlaskApp(__name__)

# app = Flask(__name__)
# app.model = HubertModel(f"{wd}/data/misc/large_noise_pt_noise_ft_433h.pt", wd)

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

@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        f = request.files['file']
        if f.filename == '':
            return
        name = time.strftime("%Y%m%d-%H%M%S")
        unique_name = f'{name}{os.path.splitext(f.filename)[-1]}'
        f.save(f'{wd}/data/video-orig/{unique_name}')
        command = f'ffmpeg -r 25 -i streamlit_app/%d.jpg -vf scale=850:480 {wd}/data/video-comp/{os.path.splitext(unique_name)[0]}_compressed.mp4'
        subprocess.call(command, shell=True)
        prediction = app.model.transcribe(unique_name)
        # prediction = "sample text"
        response = Response(prediction)
        response.headers['filename'] = name + '.txt'
        return response

@app.route('/upload_image', methods=['POST'])
def save_photo():
    # get the image data from the request
    image_data = request.json['imageString']
    image_num = request.json['imageNum']

    # decode the base-64 encoded image data
    image_data = base64.b64decode(image_data)

    # save the image data to a file
    with open(f'{wd}/streamlit_app/{image_num}.jpg', 'wb') as f:
        f.write(image_data)

    return "OK"

if __name__=='__main__':
    # app.run(host='127.0.0.1', port=8080, threaded = True, debug=True)
    app.run(host='127.0.0.1', port=8080, threaded=True)