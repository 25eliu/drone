#!/usr/bin/env python3
"""
Red Circle Landing Pad Detection
Real-time detection of bright red circular landing pads
"""

import cv2
import numpy as np
import time
from picamera2 import Picamera2
from flask import Flask, Response
import threading
import io

app = Flask(__name__)

class RedCircleDetector:
    def __init__(self):
        # Camera setup
        self.picam2 = Picamera2()
        config = self.picam2.create_video_configuration(main={"size": (640, 480)})
        self.picam2.configure(config)
        self.picam2.start()
        time.sleep(2)
        
        # Detection results
        self.latest_frame = None
        self.detection_results = {
            "detected": False,
            "center": (0, 0),
            "radius": 0,
            "area": 0,
            "distance_from_center": 0
        }
        
        # Detection parameters
        self.min_circle_area = 500      # Minimum circle area (pixels)
        self.max_circle_area = 50000    # Maximum circle area (pixels)
        self.circularity_threshold = 0.7  # How "circular" it must be (0-1)
        
        print("🎯 Red Circle Detector initialized")
        print("📹 Camera ready for landing pad detection")
    
    def detect_red_circle(self, frame):
        """
        Detect bright red circular landing pad
        Returns detection results and annotated frame
        """
        # Convert to HSV for better color detection
        hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
        
        # Define red color ranges (handles both red hues)
        # Lower red range (0-10)
        lower_red1 = np.array([0, 100, 100])
        upper_red1 = np.array([10, 255, 255])
        
        # Upper red range (160-180)
        lower_red2 = np.array([160, 100, 100])
        upper_red2 = np.array([180, 255, 255])
        
        # Create masks for both red ranges
        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        
        # Combine masks
        red_mask = cv2.bitwise_or(mask1, mask2)
        
        # Clean up the mask
        kernel = np.ones((5, 5), np.uint8)
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel)
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Frame center for distance calculation
        frame_center_x, frame_center_y = 320, 240
        
        best_circle = None
        best_score = 0
        
        # Analyze each contour
        for contour in contours:
            area = cv2.contourArea(contour)
            
            # Filter by area
            if area < self.min_circle_area or area > self.max_circle_area:
                continue
            
            # Calculate circularity
            perimeter = cv2.arcLength(contour, True)
            if perimeter == 0:
                continue
            
            circularity = 4 * np.pi * area / (perimeter * perimeter)
            
            # Filter by circularity
            if circularity < self.circularity_threshold:
                continue
            
            # Get bounding circle
            (x, y), radius = cv2.minEnclosingCircle(contour)
            center = (int(x), int(y))
            radius = int(radius)
            
            # Calculate score (prefer larger, more circular, more centered objects)
            distance_from_center = np.sqrt((x - frame_center_x)**2 + (y - frame_center_y)**2)
            
            # Scoring: larger area + higher circularity + closer to center = better
            score = area * circularity * (1 / (1 + distance_from_center/100))
            
            if score > best_score:
                best_score = score
                best_circle = {
                    "center": center,
                    "radius": radius,
                    "area": area,
                    "circularity": circularity,
                    "distance_from_center": distance_from_center,
                    "contour": contour
                }
        
        # Create annotated frame
        annotated_frame = frame.copy()
        
        if best_circle:
            # Draw detection results
            center = best_circle["center"]
            radius = best_circle["radius"]
            
            # Draw circle
            cv2.circle(annotated_frame, center, radius, (0, 255, 0), 3)
            cv2.circle(annotated_frame, center, 5, (0, 255, 0), -1)
            
            # Draw crosshairs
            cv2.line(annotated_frame, (center[0]-20, center[1]), (center[0]+20, center[1]), (0, 255, 0), 2)
            cv2.line(annotated_frame, (center[0], center[1]-20), (center[0], center[1]+20), (0, 255, 0), 2)
            
            # Draw center frame reference
            cv2.circle(annotated_frame, (frame_center_x, frame_center_y), 10, (255, 0, 0), 2)
            cv2.line(annotated_frame, (frame_center_x-15, frame_center_y), (frame_center_x+15, frame_center_y), (255, 0, 0), 2)
            cv2.line(annotated_frame, (frame_center_x, frame_center_y-15), (frame_center_x, frame_center_y+15), (255, 0, 0), 2)
            
            # Draw connection line
            cv2.line(annotated_frame, center, (frame_center_x, frame_center_y), (255, 255, 0), 1)
            
            # Add text information
            info_text = [
                f"LANDING PAD DETECTED",
                f"Center: ({center[0]}, {center[1]})",
                f"Radius: {radius}px",
                f"Area: {int(best_circle['area'])}px²",
                f"Circularity: {best_circle['circularity']:.2f}",
                f"Distance: {int(best_circle['distance_from_center'])}px",
                f"X Error: {center[0] - frame_center_x}px",
                f"Y Error: {center[1] - frame_center_y}px"
            ]
            
            for i, text in enumerate(info_text):
                color = (0, 255, 0) if i == 0 else (255, 255, 255)
                cv2.putText(annotated_frame, text, (10, 30 + i*25), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # Update detection results
            self.detection_results = {
                "detected": True,
                "center": center,
                "radius": radius,
                "area": int(best_circle['area']),
                "distance_from_center": int(best_circle['distance_from_center']),
                "x_error": center[0] - frame_center_x,
                "y_error": center[1] - frame_center_y,
                "circularity": best_circle['circularity']
            }
            
        else:
            # No detection
            cv2.putText(annotated_frame, "NO LANDING PAD DETECTED", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            cv2.putText(annotated_frame, "Searching for red circle...", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Draw center reference
            cv2.circle(annotated_frame, (frame_center_x, frame_center_y), 10, (255, 0, 0), 2)
            
            self.detection_results = {
                "detected": False,
                "center": (0, 0),
                "radius": 0,
                "area": 0,
                "distance_from_center": 0,
                "x_error": 0,
                "y_error": 0
            }
        
        return annotated_frame, self.detection_results
    
    def capture_and_detect(self):
        """Continuous capture and detection loop"""
        while True:
            try:
                # Capture frame
                frame = self.picam2.capture_array()
                
                # Detect red circle
                annotated_frame, results = self.detect_red_circle(frame)
                
                # Store latest frame for streaming
                self.latest_frame = annotated_frame
                
                # Print detection results
                if results["detected"]:
                    print(f"🎯 Landing pad at ({results['center'][0]}, {results['center'][1]}), "
                          f"Error: ({results['x_error']}, {results['y_error']}), "
                          f"Area: {results['area']}px²")
                
                time.sleep(0.1)  # 10 FPS
                
            except Exception as e:
                print(f"❌ Detection error: {e}")
                time.sleep(1)
    
    def get_latest_frame_jpeg(self):
        """Get latest annotated frame as JPEG for streaming"""
        if self.latest_frame is not None:
            _, buffer = cv2.imencode('.jpg', cv2.cvtColor(self.latest_frame, cv2.COLOR_RGB2BGR))
            return buffer.tobytes()
        return None

# Initialize detector
detector = RedCircleDetector()

def generate_frames():
    """Generate frames for Flask streaming"""
    while True:
        frame_data = detector.get_latest_frame_jpeg()
        if frame_data:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_data + b'\r\n')
        time.sleep(0.1)

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), 
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/detection_data')
def detection_data():
    """API endpoint for detection results"""
    return detector.detection_results

@app.route('/')
def index():
    return '''
    <html>
    <head>
        <title>🎯 Red Circle Landing Pad Detection</title>
        <style>
            body { 
                font-family: Arial; 
                background: #1a1a1a; 
                color: white; 
                text-align: center; 
                margin: 0; 
                padding: 20px;
            }
            .container { 
                max-width: 900px; 
                margin: 0 auto; 
            }
            .video-container {
                background: #333;
                padding: 20px;
                border-radius: 10px;
                margin: 20px 0;
            }
            .detection-info {
                background: #2a2a2a;
                padding: 15px;
                border-radius: 8px;
                margin: 10px 0;
                text-align: left;
            }
            .status-indicator {
                width: 20px;
                height: 20px;
                border-radius: 50%;
                display: inline-block;
                margin-right: 10px;
            }
            .detected { background: #00ff00; }
            .not-detected { background: #ff0000; }
        </style>
        <script>
            function updateDetectionInfo() {
                fetch('/detection_data')
                    .then(response => response.json())
                    .then(data => {
                        const status = document.getElementById('status');
                        const indicator = document.getElementById('indicator');
                        const info = document.getElementById('info');
                        
                        if (data.detected) {
                            status.innerText = 'LANDING PAD DETECTED';
                            indicator.className = 'status-indicator detected';
                            info.innerHTML = `
                                <strong>Position:</strong> (${data.center[0]}, ${data.center[1]})<br>
                                <strong>Radius:</strong> ${data.radius}px<br>
                                <strong>Area:</strong> ${data.area}px²<br>
                                <strong>X Error:</strong> ${data.x_error}px<br>
                                <strong>Y Error:</strong> ${data.y_error}px<br>
                                <strong>Distance from Center:</strong> ${data.distance_from_center}px
                            `;
                        } else {
                            status.innerText = 'SEARCHING FOR LANDING PAD';
                            indicator.className = 'status-indicator not-detected';
                            info.innerHTML = '<em>No red circle detected</em>';
                        }
                    });
            }
            
            setInterval(updateDetectionInfo, 500);
        </script>
    </head>
    <body>
        <div class="container">
            <h1>🎯 Red Circle Landing Pad Detection</h1>
            <h2>🚁 Pi Zero 2W Drone Vision System</h2>
            
            <div class="video-container">
                <img src="/video_feed" width="640" height="480" style="border: 2px solid #555;">
            </div>
            
            <div class="detection-info">
                <h3>
                    <span id="indicator" class="status-indicator not-detected"></span>
                    <span id="status">INITIALIZING...</span>
                </h3>
                <div id="info">Loading detection system...</div>
            </div>
            
            <div class="detection-info">
                <h4>🎯 Detection Parameters:</h4>
                <ul style="text-align: left;">
                    <li><strong>Color:</strong> Bright red (HSV ranges)</li>
                    <li><strong>Shape:</strong> Circular (circularity > 0.7)</li>
                    <li><strong>Size:</strong> 500-50000 pixels²</li>
                    <li><strong>Target:</strong> Center of frame (320, 240)</li>
                </ul>
            </div>
            
            <div class="detection-info">
                <h4>🔧 Usage Instructions:</h4>
                <ol style="text-align: left;">
                    <li>Create a bright red circular landing pad</li>
                    <li>Ensure high contrast with background</li>
                    <li>Point camera at the landing pad</li>
                    <li>Green circle = detected, red cross = frame center</li>
                    <li>Minimize X/Y error for precise landing</li>
                </ol>
            </div>
        </div>
    </body>
    </html>
    '''

if __name__ == '__main__':
    print("🎯 Starting Red Circle Landing Pad Detection...")
    print("📹 Camera initializing...")
    
    # Start detection thread
    detection_thread = threading.Thread(target=detector.capture_and_detect)
    detection_thread.daemon = True
    detection_thread.start()
    
    print("🌐 Web interface starting...")
    print("📺 Access detection at: http://raspberrypi.local:5000")
    print("🎯 Point camera at a bright red circular object")
    print("🛑 Press Ctrl+C to stop")
    
    try:
        app.run(host='0.0.0.0', port=5000, threaded=True, debug=False)
    finally:
        detector.picam2.stop()
        detector.picam2.close()
        print("📷 Camera stopped")