#!/usr/bin/env python3
"""
Flask Camera Streaming for Pi Zero 2W
"""

import io
import time
from flask import Flask, Response
from picamera2 import Picamera2

app = Flask(__name__)

# Global camera object
picam2 = None

def init_camera():
    global picam2
    if picam2 is None:
        picam2 = Picamera2()
        config = picam2.create_video_configuration(main={"size": (640, 480)})
        picam2.configure(config)
        picam2.start()
        time.sleep(2)  # Camera warm-up

def generate_frames():
    init_camera()
    
    while True:
        try:
            # Capture frame as JPEG
            stream = io.BytesIO()
            picam2.capture_file(stream, format='jpeg')
            stream.seek(0)
            
            # Yield frame for streaming
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + stream.read() + b'\r\n')
            
        except Exception as e:
            print(f"Camera error: {e}")
            time.sleep(1)

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), 
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    return '''
    <html>
    <head>
        <title>🚁 Pi Zero 2W Drone Camera</title>
        <style>
            body { font-family: Arial; text-align: center; background: #f0f0f0; }
            .container { max-width: 800px; margin: 0 auto; padding: 20px; }
            .camera-feed { border: 3px solid #333; border-radius: 10px; }
            h1 { color: #333; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🚁 Pi Zero 2W Drone Camera Stream</h1>
            <h2>📹 Live Camera Feed</h2>
            <img src="/video_feed" class="camera-feed" width="640" height="480">
            <p><strong>✅ Camera Status:</strong> Streaming at 640x480</p>
            <p><strong>🔗 Stream URL:</strong> /video_feed</p>
            <p><strong>🚁 Ready for drone integration!</strong></p>
        </div>
    </body>
    </html>
    '''

if __name__ == '__main__':
    print("🚁 Starting Pi Zero 2W Drone Camera Stream...")
    print("📺 Access stream at: http://raspberrypi.local:5000")
    print("🛑 Press Ctrl+C to stop")
    
    try:
        app.run(host='0.0.0.0', port=5000, threaded=True, debug=False)
    finally:
        if picam2:
            picam2.stop()
            picam2.close()

