const FRAMERATE = 25
var video = document.querySelector('video');
var textfiles = document.querySelector('.textfiles')
var filelist = document.querySelector('.items-body')
var recorder;
var outputs = [];
let counter = 0;
let intervalId = 0
let batch = -1

var show = (elem) => {
    elem.style.display = 'block';
};
var hide = (elem) => {
    elem.style.display = 'none';
};

function captureCamera(callback) {
    navigator.mediaDevices.getUserMedia({ audio: true, video: true }).then(function(camera) {
        callback(camera);
    }).catch(function(error) {
        alert('Unable to capture your camera. Please check console logs.');
        console.error(error);
    });
}

function stopRecordingCallback() {
    video.src = URL.createObjectURL(recorder.getBlob());
    hide(video);
    show(textfiles);
    
    // Unmute video
    video.muted = false;

    // Stop device streaming
    recorder.camera.stop()

    let recording = new File([recorder.getBlob()], 'recording.webm', {type: 'video/webm'});
    uploadFile(recording);

    // Destroy original recorder
    recorder.destroy();
    recorder = null;

    // Enable record button again
    document.getElementById('btn-start-recording').disabled = false;
}

function updateBatch() {
    batch = Date.now();
}

function generateTxtFile(text) {
    var textFile = null;
    var data = new Blob([text], {type: 'text/plain'}); 
    if (textFile !== null) {  
      window.URL.revokeObjectURL(textFile);  
    }  
    textFile = window.URL.createObjectURL(data);  
    return textFile; 
}

function updateList(index) {
    let rowData = outputs[index];
    let filename = rowData['filename'];
    let prediction = rowData['prediction'];
    let filepath = rowData['filepath'];

    filelist.innerHTML += "<div class='items-body-row'>" +
        "<span class='text-preview'><h3 class='file-name'>" +
        filename + "</h3>" + 
        prediction + "</span><a href='" +
        filepath + "' download=" +
        filename + "><div class='btn download-btn'>" +
        "<i class='fa fa-download'></i></div></a></div>"
}

hide(video);
show(textfiles);

document.getElementById('btn-start-recording').onclick = function() {
    this.disabled = true;
    updateBatch();

    captureCamera(function(camera) {
        video.muted = true;
        // video.volume = 0;
        video.srcObject = camera;

        recorder = RecordRTC(camera, {
            type: 'video',
            canvas: {width: 640, height: 480},
        });

        var mediaContainer = document.querySelector('.media-player');
        video.height = mediaContainer.offsetHeight;

        hide(textfiles);
        show(video);
        recorder.startRecording();

        // release camera on stopRecording
        recorder.camera = camera;

        document.getElementById('btn-stop-recording').disabled = false;
    });

    console.log("Starting interval")
    intervalId = setInterval(captureFrame, 1000 / FRAMERATE)
};

document.getElementById('btn-stop-recording').onclick = function() {
    this.disabled = true;
    recorder.stopRecording(stopRecordingCallback);
    console.log("Clearing interval");
    clearInterval(intervalId);
    counter = 0;
};

var canvas = document.createElement("canvas");
canvas.width = 320
canvas.height = 240
var ctx = canvas.getContext("2d");
function captureFrame() {
    if (recorder) {
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height)
        var imageData = canvas.toDataURL("image/jpeg").replace(/^data:image\/(png|jpg|jpeg);base64,/, "");
        uploadImage(imageData, counter)
        counter += 1
    }
}

async function uploadImage(imageData, counter) {
    let response = await fetch('/upload_image', {
        method: "POST",
        headers: {
            "Content-Type": "application/json",
        },
        body: JSON.stringify({
            imageString: imageData,
            imageNum: counter,
            batchName: batch,
        })
    })
}

async function uploadFile(recording) {
        let data = new FormData(); 
        data.append("file", recording);
        data.append("batchName", batch)
        let response = await fetch('/upload', {
            method: "POST", 
            body: data
        }); 

        prediction = await response.text()
        filename = response.headers.get('filename')

        console.log(prediction)
        console.log(filename)

        if (prediction.length > 0) {
            let txtFile = generateTxtFile(prediction)
            outputs.push({
                "filename": filename,
                "prediction": prediction, 
                "filepath": txtFile
            })

            updateList(outputs.length - 1)
        }
        
        console.log(outputs)
};
