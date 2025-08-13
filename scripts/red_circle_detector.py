#!/usr/bin/env python3
"""
red_circle_detector.py
Standalone red circle detection module with Flask visualization
Can be imported by other modules or run independently with web interface
"""

import cv2
import numpy as np
import time
import threading
from dataclasses import dataclass
from typing import Tuple, Dict, Optional
from flask import Flask, Response, render_template_string
import json

@dataclass
class DetectionResult:
    """Data class for detection results"""
    detected: bool
    center: Tuple[int, int]
    radius: int
    area: int
    distance_from_center: float
    x_error: int
    y_error: int
    circularity: float = 0.0
    confidence: float = 0.0

class RedCircleDetector:
    def __init__(self, use_camera=True, frame_size=(640, 480)):
        """
        Initialize detector
        Args:
            use_camera: If True, use real camera. If False, use for simulation
            frame_size: Tuple of (width, height)
        """
        self.use_camera = use_camera
        self.frame_size = frame_size
        self.frame_center = (frame_size[0]//2, frame_size[1]//2)
        
        # Camera setup (only if using real camera)
        self.picam2 = None
        if use_camera:
            try:
                from picamera2 import Picamera2
                self.picam2 = Picamera2()
                config = self.picam2.create_video_configuration(
                    main={"size": frame_size}
                )
                self.picam2.configure(config)
                self.picam2.start()
                time.sleep(2)
                print("📷 Camera initialized successfully")
            except ImportError:
                print("⚠️ Picamera2 not available, running in simulation mode")
                self.use_camera = False
            except Exception as e:
                print(f"⚠️ Camera initialization failed: {e}")
                self.use_camera = False
        
        # Detection parameters
        self.min_circle_area = 500
        self.max_circle_area = 50000
        self.circularity_threshold = 0.7
        
        # Thread safety
        self.lock = threading.Lock()
        self.latest_frame = None
        self.latest_annotated_frame = None
        self.latest_result = DetectionResult(
            detected=False,
            center=(0, 0),
            radius=0,
            area=0,
            distance_from_center=0,
            x_error=0,
            y_error=0
        )
        
        print("🎯 Red Circle Detector initialized")
    
    def create_simulated_frame(self, circle_pos=None, add_noise=True):
        """
        Create a simulated frame for testing without camera
        Args:
            circle_pos: Tuple (x, y) for circle center, None for random
            add_noise: Add random noise to simulate real conditions
        """
        # Create blank frame
        frame = np.ones((self.frame_size[1], self.frame_size[0], 3), dtype=np.uint8) * 200
        
        if circle_pos is None:
            # Random position
            x = np.random.randint(100, self.frame_size[0] - 100)
            y = np.random.randint(100, self.frame_size[1] - 100)
        else:
            x, y = circle_pos
        
        # Random radius
        radius = np.random.randint(30, 80)
        
        # Draw red circle
        cv2.circle(frame, (x, y), radius, (255, 0, 0), -1)  # BGR format
        
        # Add noise if requested
        if add_noise:
            noise = np.random.randint(-20, 20, frame.shape, dtype=np.int16)
            frame = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
        # Convert BGR to RGB for processing
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        return frame
    
    def capture_frame(self):
        """Capture a frame from camera or simulation"""
        if self.use_camera and self.picam2:
            return self.picam2.capture_array()
        else:
            # Return simulated frame
            return self.create_simulated_frame()
    
    def detect_red_circle(self, frame):
        """
        Detect red circular landing pad in frame
        Returns: DetectionResult object
        """
        # Convert to HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
        
        # Red color ranges
        lower_red1 = np.array([0, 100, 100])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([160, 100, 100])
        upper_red2 = np.array([180, 255, 255])
        
        # Create masks
        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        red_mask = cv2.bitwise_or(mask1, mask2)
        
        # Clean up mask
        kernel = np.ones((5, 5), np.uint8)
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel)
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(
            red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        
        best_circle = None
        best_score = 0
        
        for contour in contours:
            area = cv2.contourArea(contour)
            
            if area < self.min_circle_area or area > self.max_circle_area:
                continue
            
            perimeter = cv2.arcLength(contour, True)
            if perimeter == 0:
                continue
            
            circularity = 4 * np.pi * area / (perimeter * perimeter)
            
            if circularity < self.circularity_threshold:
                continue
            
            (x, y), radius = cv2.minEnclosingCircle(contour)
            center = (int(x), int(y))
            radius = int(radius)
            
            # Calculate distance from frame center
            distance_from_center = np.sqrt(
                (x - self.frame_center[0])**2 + 
                (y - self.frame_center[1])**2
            )
            
            # Scoring
            score = area * circularity * (1 / (1 + distance_from_center/100))
            
            if score > best_score:
                best_score = score
                best_circle = {
                    "center": center,
                    "radius": radius,
                    "area": area,
                    "circularity": circularity,
                    "distance_from_center": distance_from_center
                }
        
        if best_circle:
            return DetectionResult(
                detected=True,
                center=best_circle["center"],
                radius=best_circle["radius"],
                area=int(best_circle["area"]),
                distance_from_center=best_circle["distance_from_center"],
                x_error=best_circle["center"][0] - self.frame_center[0],
                y_error=best_circle["center"][1] - self.frame_center[1],
                circularity=best_circle["circularity"],
                confidence=min(best_circle["circularity"], 1.0)
            )
        else:
            return DetectionResult(
                detected=False,
                center=(0, 0),
                radius=0,
                area=0,
                distance_from_center=0,
                x_error=0,
                y_error=0,
                circularity=0,
                confidence=0
            )
    
    def get_latest_detection(self) -> DetectionResult:
        """Thread-safe getter for latest detection result"""
        with self.lock:
            return self.latest_result
    
    def annotate_frame(self, frame, detection_result):
        """Add visualization overlays to frame"""
        annotated = frame.copy()
        
        # Draw frame center
        cv2.circle(annotated, self.frame_center, 10, (255, 0, 0), 2)
        cv2.line(annotated, 
                (self.frame_center[0]-15, self.frame_center[1]),
                (self.frame_center[0]+15, self.frame_center[1]),
                (255, 0, 0), 2)
        cv2.line(annotated,
                (self.frame_center[0], self.frame_center[1]-15),
                (self.frame_center[0], self.frame_center[1]+15),
                (255, 0, 0), 2)
        
        if detection_result.detected:
            # Draw detected circle
            cv2.circle(annotated, detection_result.center, 
                      detection_result.radius, (0, 255, 0), 3)
            cv2.circle(annotated, detection_result.center, 5, (0, 255, 0), -1)
            
            # Draw crosshairs on target
            cx, cy = detection_result.center
            cv2.line(annotated, (cx-20, cy), (cx+20, cy), (0, 255, 0), 2)
            cv2.line(annotated, (cx, cy-20), (cx, cy+20), (0, 255, 0), 2)
            
            # Draw connection line
            cv2.line(annotated, detection_result.center, 
                    self.frame_center, (255, 255, 0), 1)
            
            # Add text
            texts = [
                "LANDING PAD DETECTED",
                f"Center: {detection_result.center}",
                f"Radius: {detection_result.radius}px",
                f"Area: {detection_result.area}px²",
                f"Circularity: {detection_result.circularity:.2f}",
                f"Distance: {int(detection_result.distance_from_center)}px",
                f"X Error: {detection_result.x_error}px",
                f"Y Error: {detection_result.y_error}px"
            ]
            
            for i, text in enumerate(texts):
                color = (0, 255, 0) if i == 0 else (255, 255, 255)
                cv2.putText(annotated, text, (10, 30 + i*25),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        else:
            cv2.putText(annotated, "NO LANDING PAD DETECTED", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            cv2.putText(annotated, "Searching for red circle...", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return annotated
    
    def continuous_detection(self, callback=None):
        """
        Run continuous detection loop
        Args:
            callback: Optional function to call with each detection result
        """
        while True:
            try:
                frame = self.capture_frame()
                result = self.detect_red_circle(frame)
                annotated_frame = self.annotate_frame(frame, result)
                
                with self.lock:
                    self.latest_frame = frame
                    self.latest_annotated_frame = annotated_frame
                    self.latest_result = result
                
                if callback:
                    callback(result)
                
                time.sleep(0.05)  # 20 FPS
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"Detection error: {e}")
                time.sleep(1)
    
    def get_latest_frame_jpeg(self):
        """Get latest annotated frame as JPEG for streaming"""
        with self.lock:
            if self.latest_annotated_frame is not None:
                _, buffer = cv2.imencode('.jpg', cv2.cvtColor(self.latest_annotated_frame, cv2.COLOR_RGB2BGR))
                return buffer.tobytes()
        return None
    
    def cleanup(self):
        """Clean up resources"""
        if self.picam2:
            self.picam2.stop()
            self.picam2.close()
            print("Camera closed")


# Flask app for standalone testing
app = Flask(__name__)
detector = None

def generate_frames():
    """Generate frames for Flask streaming"""
    while True:
        if detector:
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
    if detector:
        result = detector.get_latest_detection()
        return {
            "detected": result.detected,
            "center": result.center,
            "radius": result.radius,
            "area": result.area,
            "distance_from_center": result.distance_from_center,
            "x_error": result.x_error,
            "y_error": result.y_error,
            "circularity": result.circularity,
            "confidence": result.confidence
        }
    return {"error": "No detector initialized"}

@app.route('/')
def index():
    mode_text = "REAL CAMERA" if detector.use_camera else "SIMULATION MODE"
    return render_template_string('''
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
                position: relative;
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
            .detected { 
                background: #00ff00; 
                box-shadow: 0 0 10px #00ff00;
                animation: pulse 1s infinite;
            }
            .not-detected { 
                background: #ff0000; 
                box-shadow: 0 0 10px #ff0000;
            }
            @keyframes pulse {
                0% { opacity: 1; }
                50% { opacity: 0.5; }
                100% { opacity: 1; }
            }
            .mode-badge {
                display: inline-block;
                background: #007bff;
                padding: 5px 15px;
                border-radius: 20px;
                margin: 10px;
                font-weight: bold;
            }
            .metrics {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
                gap: 10px;
                margin-top: 15px;
            }
            .metric-card {
                background: rgba(255, 255, 255, 0.1);
                padding: 10px;
                border-radius: 5px;
            }
            .metric-value {
                font-size: 24px;
                font-weight: bold;
                color: #4fc3f7;
            }
            .metric-label {
                font-size: 12px;
                color: #888;
                margin-top: 5px;
            }
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
                            
                            // Update metrics
                            document.getElementById('position').innerText = `(${data.center[0]}, ${data.center[1]})`;
                            document.getElementById('x-error').innerText = `${data.x_error}px`;
                            document.getElementById('y-error').innerText = `${data.y_error}px`;
                            document.getElementById('radius').innerText = `${data.radius}px`;
                            document.getElementById('area').innerText = `${data.area}px²`;
                            document.getElementById('distance').innerText = `${Math.round(data.distance_from_center)}px`;
                            document.getElementById('circularity').innerText = `${(data.circularity * 100).toFixed(1)}%`;
                            document.getElementById('confidence').innerText = `${(data.confidence * 100).toFixed(1)}%`;
                        } else {
                            status.innerText = 'SEARCHING FOR LANDING PAD';
                            indicator.className = 'status-indicator not-detected';
                            
                            // Clear metrics
                            document.getElementById('position').innerText = '-';
                            document.getElementById('x-error').innerText = '-';
                            document.getElementById('y-error').innerText = '-';
                            document.getElementById('radius').innerText = '-';
                            document.getElementById('area').innerText = '-';
                            document.getElementById('distance').innerText = '-';
                            document.getElementById('circularity').innerText = '-';
                            document.getElementById('confidence').innerText = '-';
                        }
                    });
            }
            
            setInterval(updateDetectionInfo, 100);
            updateDetectionInfo();
        </script>
    </head>
    <body>
        <div class="container">
            <h1>🎯 Red Circle Landing Pad Detection</h1>
            <div class="mode-badge">{{ mode }}</div>
            
            <div class="video-container">
                <img src="/video_feed" width="640" height="480" style="border: 2px solid #555;">
            </div>
            
            <div class="detection-info">
                <h3>
                    <span id="indicator" class="status-indicator not-detected"></span>
                    <span id="status">INITIALIZING...</span>
                </h3>
                
                <div class="metrics">
                    <div class="metric-card">
                        <div class="metric-value" id="position">-</div>
                        <div class="metric-label">CENTER POSITION</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value" id="x-error">-</div>
                        <div class="metric-label">X ERROR</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value" id="y-error">-</div>
                        <div class="metric-label">Y ERROR</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value" id="radius">-</div>
                        <div class="metric-label">RADIUS</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value" id="area">-</div>
                        <div class="metric-label">AREA</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value" id="distance">-</div>
                        <div class="metric-label">DISTANCE</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value" id="circularity">-</div>
                        <div class="metric-label">CIRCULARITY</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value" id="confidence">-</div>
                        <div class="metric-label">CONFIDENCE</div>
                    </div>
                </div>
            </div>
            
            <div class="detection-info">
                <h4>🎯 Detection Parameters:</h4>
                <ul style="text-align: left;">
                    <li><strong>Color:</strong> Bright red (HSV ranges: 0-10, 160-180)</li>
                    <li><strong>Shape:</strong> Circular (circularity > 0.7)</li>
                    <li><strong>Size:</strong> 500-50000 pixels²</li>
                    <li><strong>Target:</strong> Center of frame (320, 240)</li>
                    <li><strong>Mode:</strong> {{ mode }}</li>
                </ul>
            </div>
            
            <div class="detection-info">
                <h4>ℹ️ Visual Indicators:</h4>
                <ul style="text-align: left;">
                    <li>🟢 <strong>Green Circle:</strong> Detected landing pad boundary</li>
                    <li>🔵 <strong>Blue Cross:</strong> Frame center (target position)</li>
                    <li>🟡 <strong>Yellow Line:</strong> Connection from pad to center</li>
                    <li>✅ <strong>Green Crosshair:</strong> Landing pad center</li>
                </ul>
            </div>
        </div>
    </body>
    </html>
    ''', mode=mode_text)


# Standalone testing
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Red Circle Detector")
    parser.add_argument('--no-camera', action='store_true', 
                       help='Run in simulation mode without camera')
    parser.add_argument('--web', action='store_true',
                       help='Start web interface for visualization')
    parser.add_argument('--port', type=int, default=5000,
                       help='Port for web interface (default: 5000)')
    args = parser.parse_args()
    
    print("🎯 Red Circle Detector Standalone Test")
    print("=" * 50)
    
    # Initialize detector
    use_camera = not args.no_camera
    detector = RedCircleDetector(use_camera=use_camera)
    
    if args.web:
        # Web interface mode
        print(f"🌐 Starting web interface on port {args.port}")
        print(f"📺 Access detection at: http://localhost:{args.port}")
        print(f"📷 Mode: {'REAL CAMERA' if use_camera else 'SIMULATION'}")
        print("🛑 Press Ctrl+C to stop")
        
        # Start detection thread
        detection_thread = threading.Thread(target=detector.continuous_detection)
        detection_thread.daemon = True
        detection_thread.start()
        
        try:
            app.run(host='0.0.0.0', port=args.port, debug=False)
        except KeyboardInterrupt:
            print("\n✋ Shutting down...")
        finally:
            detector.cleanup()
    else:
        # Console mode
        print(f"📷 Mode: {'REAL CAMERA' if use_camera else 'SIMULATION'}")
        print("Running detection for 30 seconds...")
        print("(Use --web flag to see visual output)")
        print("-" * 50)
        
        def detection_callback(result):
            if result.detected:
                print(f"✅ Detected at ({result.center[0]}, {result.center[1]}) "
                      f"Error: ({result.x_error:+4d}, {result.y_error:+4d}) "
                      f"Confidence: {result.confidence:.2f}")
            else:
                print("❌ No detection")
        
        # Run for 30 seconds
        start_time = time.time()
        detection_thread = threading.Thread(
            target=lambda: detector.continuous_detection(callback=detection_callback)
        )
        detection_thread.daemon = True
        detection_thread.start()
        
        try:
            while time.time() - start_time < 30:
                time.sleep(1)
                remaining = 30 - int(time.time() - start_time)
                if remaining % 5 == 0 and remaining > 0:
                    print(f"⏱️ {remaining} seconds remaining...")
        except KeyboardInterrupt:
            print("\n✋ Interrupted")
        
        detector.cleanup()
        print("-" * 50)
        print("Test completed!")