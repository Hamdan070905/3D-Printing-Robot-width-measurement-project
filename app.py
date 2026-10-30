from flask import Flask, render_template_string, request, Response
import cv2
from ultralytics import YOLO
import numpy as np
import base64

app = Flask(__name__)

# Load your trained train-4 model
MODEL_PATH = r"D:\New width prj\runs\segment\train-4\weights\best.pt"
model = YOLO(MODEL_PATH)

# Frontend HTML with Mobile Camera Access via JavaScript
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>3DCP Mobile Live Metrology</title>
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        body { font-family: Arial; text-align: center; background: #12161c; color: #fff; margin: 0; padding: 15px; }
        h2 { color: #50fa7b; font-size: 20px; }
        video { display: none; }
        canvas { display: none; }
        .video-box { margin-top: 15px; border: 2px solid #50fa7b; display: inline-block; border-radius: 8px; overflow: hidden; max-width: 100%; }
        #output { width: 100%; max-width: 480px; height: auto; display: block; }
        .status { margin-top: 10px; font-size: 14px; color: #8be9fd; }
    </style>
</head>
<body>
    <h2>3D Concrete Printing Live Metrology</h2>
    <div class="status" id="status">Connecting phone camera...</div>
    
    <div class="video-box">
        <img id="output" alt="Waiting for AI stream..." />
    </div>

    <video id="video" autoplay playsinline></video>
    <canvas id="canvas"></canvas>

    <script>
        const video = document.getElementById('video');
        const canvas = document.getElementById('canvas');
        const context = canvas.getContext('2d');
        const output = document.getElementById('output');
        const status = document.getElementById('status');

        navigator.mediaDevices.getUserMedia({ video: { facingMode: "environment" }, audio: false })
            .then(stream => {
                video.srcObject = stream;
                status.innerText = "Camera Active - Streaming to Laptop AI...";
                video.play();
                requestAnimationFrame(sendFrame);
            })
            .catch(err => {
                status.innerText = "Camera Error: " + err;
            });

        function sendFrame() {
            if (video.videoWidth === 0) {
                setTimeout(sendFrame, 200);
                return;
            }
            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;
            context.drawImage(video, 0, 0, canvas.width, canvas.height);
            
            const dataURL = canvas.toDataURL('image/jpeg', 0.5);

            fetch('/process_frame', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ image: dataURL })
            })
            .then(response => response.json())
            .then(data => {
                output.src = 'data:image/jpeg;base64,' + data.image;
                requestAnimationFrame(sendFrame);
            })
            .catch(err => {
                setTimeout(sendFrame, 500);
            });
        }
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/process_frame', methods=['POST'])
def process_frame():
    data = request.get_json()
    if not data or 'image' not in data:
        return {'image': ''}

    img_data = data['image'].split(',')[1]
    np_bytes = np.frombuffer(base64.b64decode(img_data), np.uint8)
    frame = cv2.imdecode(np_bytes, cv2.IMREAD_COLOR)

    if frame is not None:
        results = model.track(source=frame, persist=True, verbose=False)
        annotated_frame = results[0].plot() if results else frame
    else:
        annotated_frame = frame

    # FIXED: Changed from IMREAD to IMWRITE
    _, buffer = cv2.imencode('.jpg', annotated_frame, [cv2.IMWRITE_JPEG_QUALITY, 60])
    encoded_image = base64.b64encode(buffer).decode('utf-8')

    return {'image': encoded_image}

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)