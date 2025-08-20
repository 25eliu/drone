#!/usr/bin/env python3
import math
import time
import threading
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np
from pymavlink import mavutil

from red_circle_detector import RedCircleDetector, DetectionResult

@dataclass
class CameraModel:
    width: int = 640
    height: int = 480
    hfov_deg: float = 78.0   # <-- set to your lens HFOV
    vfov_deg: Optional[float] = None  # if None, computed from aspect ratio

    def __post_init__(self):
        if self.vfov_deg is None:
            # derive VFOV from HFOV and aspect ratio
            aspect = self.height / self.width
            hfov = math.radians(self.hfov_deg)
            # approximate vfov via tan relation
            self.vfov_deg = math.degrees(2 * math.atan(math.tan(hfov/2) * aspect))
        # intrinsics (approx)
        self.fx = (self.width / 2.0) / math.tan(math.radians(self.hfov_deg) / 2.0)
        self.fy = (self.height / 2.0) / math.tan(math.radians(self.vfov_deg) / 2.0)
        self.cx = self.width // 2
        self.cy = self.height // 2

class LandingTargetPublisher:
    def __init__(self,
                 #mavlink_url: str = "udpout:127.0.0.1:14550", (for wirelesss setup)
                 mavlink_url: str = "serial:/dev/serial0:115200",
                 camera: CameraModel = CameraModel(),
                 target_size_m: float = 0.3,    # diameter of the red circle (meters), optional
                 yaw_align_rad: float = 0.0,    # yaw of camera w.r.t. body (usually 0 if forward)
                 verbose_qgc: bool = True):      # Enable QGC status messages
        """
        mavlink_url examples:
          - "serial:/dev/ttyAMA0:921600"
          - "udpin:0.0.0.0:14550"
          - "udpout:127.0.0.1:14550"
          - "tcp:127.0.0.1:5760"
        """
        self.master = mavutil.mavlink_connection(mavlink_url, autoreconnect=True)
        self.camera = camera
        self.target_size_m = target_size_m
        self.yaw_align_rad = yaw_align_rad
        self.verbose_qgc = verbose_qgc
        self.heartbeat_wait()

        # Create the detector and start it
        self.detector = RedCircleDetector(use_camera=True, frame_size=(camera.width, camera.height))
        self.running = True

        # statistics
        self._last_send = 0.0
        self._stable_hits = 0
        self._last_status_time = 0.0
        self._status_interval = 5.0  # Send status every 5 seconds
        self._detection_state = False  # Track detection state changes
        self._total_detections = 0
        self._startup_time = time.time()
        
        # Send startup message
        self.send_qgc_message("PrecLand: Starting vision system", severity=mavutil.mavlink.MAV_SEVERITY_INFO)

    def heartbeat_wait(self):
        print("Waiting for FCU heartbeat...")
        msg = self.master.wait_heartbeat(timeout=10)
        if not msg:
            self.send_qgc_message("PrecLand: No heartbeat from FCU!", severity=mavutil.mavlink.MAV_SEVERITY_ERROR)
            raise Exception("No heartbeat received - check connection")
        print("✅ Heartbeat received")
        self.send_qgc_message("PrecLand: Connected to FCU", severity=mavutil.mavlink.MAV_SEVERITY_INFO)

    def send_qgc_message(self, text, severity=mavutil.mavlink.MAV_SEVERITY_INFO):
        """
        Send status text message that appears in QGC
        Severity levels:
        - MAV_SEVERITY_EMERGENCY = 0
        - MAV_SEVERITY_ALERT = 1
        - MAV_SEVERITY_CRITICAL = 2  
        - MAV_SEVERITY_ERROR = 3
        - MAV_SEVERITY_WARNING = 4
        - MAV_SEVERITY_NOTICE = 5
        - MAV_SEVERITY_INFO = 6
        - MAV_SEVERITY_DEBUG = 7
        """
        try:
            # Truncate to 50 chars (MAVLink limit)
            text = text[:50]
            self.master.mav.statustext_send(
                severity,
                text.encode('utf-8')
            )
            print(f"[QGC MSG] {text}")
        except Exception as e:
            print(f"Failed to send QGC message: {e}")

    def to_angles(self, result: DetectionResult):
        # Pixel errors
        du = result.x_error
        dv = result.y_error
        # Convert to small-angle approximation (or exact atan)
        x_angle = math.atan(du / self.camera.fx)
        y_angle = math.atan(dv / self.camera.fy)
        return x_angle, y_angle

    def estimate_distance(self, radius_pixels):
        """Estimate distance from target size (optional)"""
        if radius_pixels > 0:
            focal_length = self.camera.fx
            actual_radius = self.target_size_m / 2.0
            distance = (actual_radius * focal_length) / radius_pixels
            return distance
        return None

    def send_landing_target(self, x_angle, y_angle, distance_m=None, frame=mavutil.mavlink.MAV_FRAME_BODY_NED):
        # If you have a rangefinder, pass its distance here (improves EKF)
        # Time (usec)
        ts = int(time.time() * 1e6)
        self.master.mav.landing_target_send(
            int(time.time()*1e6) & 0xFFFFFFFFFFFFFFFF,  # time_usec, uint64
            0,                                          # target_num
            int(frame),                                 # frame
            float(x_angle),
            float(y_angle),
            float(distance_m or 0.0),
            0.0,
            0.0,
            2,                                          # type
            1                                           # position_valid
        )

    def send_periodic_status(self, result: DetectionResult):
        """Send periodic status updates to QGC"""
        now = time.time()
        
        # Send status every N seconds when detected
        if now - self._last_status_time > self._status_interval:
            if result.detected:
                # Send detailed status when target is detected
                msg = f"PrecLand: Target@({result.x_error:+3.0f},{result.y_error:+3.0f})px"
                self.send_qgc_message(msg, severity=mavutil.mavlink.MAV_SEVERITY_INFO)
                
                # Optionally send confidence
                if result.confidence > 0:
                    conf_msg = f"PrecLand: Confidence {result.confidence*100:.0f}%"
                    self.send_qgc_message(conf_msg, severity=mavutil.mavlink.MAV_SEVERITY_DEBUG)
            else:
                # Periodic "still searching" message
                uptime = int(now - self._startup_time)
                msg = f"PrecLand: Searching... (up {uptime}s)"
                self.send_qgc_message(msg, severity=mavutil.mavlink.MAV_SEVERITY_INFO)
            
            self._last_status_time = now
        
        # Send immediate message on detection state change
        if result.detected != self._detection_state:
            self._detection_state = result.detected
            if result.detected:
                self._total_detections += 1
                msg = f"PrecLand: TARGET ACQUIRED #{self._total_detections}"
                self.send_qgc_message(msg, severity=mavutil.mavlink.MAV_SEVERITY_NOTICE)
            else:
                msg = "PrecLand: Target lost"
                self.send_qgc_message(msg, severity=mavutil.mavlink.MAV_SEVERITY_WARNING)

    def maybe_switch_to_land(self):
        # Optional: if target stable for N cycles, command LAND
        if self._stable_hits >= 10:  # ~0.5s at 20 Hz
            try:
                # guided → land (safe sequence), but you can just LAND directly
                self.master.set_mode("LAND")
                print("🛬 LAND command sent (Precision Land engaged)")
                self.send_qgc_message("PrecLand: AUTO-LAND activated", severity=mavutil.mavlink.MAV_SEVERITY_NOTICE)
                self._stable_hits = 0
            except Exception as e:
                print(f"Mode change failed: {e}")
                self.send_qgc_message(f"PrecLand: Mode change failed", severity=mavutil.mavlink.MAV_SEVERITY_ERROR)

    def run(self):
        # Background heartbeat sender (important for some links)
        def hb():
            while self.running:
                try:
                    self.master.mav.heartbeat_send(
                        mavutil.mavlink.MAV_TYPE_ONBOARD_CONTROLLER,
                        mavutil.mavlink.MAV_AUTOPILOT_INVALID, 0, 0, 0
                    )
                except Exception:
                    pass
                time.sleep(1)

        threading.Thread(target=hb, daemon=True).start()

        # Start detector loop in own thread (keeps your Flask UI working if you launch it)
        threading.Thread(target=self.detector.continuous_detection, daemon=True).start()

        print("🔁 Streaming LANDING_TARGET at ~20 Hz when target is visible…")
        self.send_qgc_message("PrecLand: System ready", severity=mavutil.mavlink.MAV_SEVERITY_INFO)
        
        try:
            while True:
                res = self.detector.get_latest_detection()
                now = time.time()
                
                # Send periodic status to QGC
                if self.verbose_qgc:
                    self.send_periodic_status(res)
                
                if res.detected and (now - self._last_send) > 0.05:  # 20 Hz
                    x_angle, y_angle = self.to_angles(res)
                    
                    # Optionally estimate distance from circle size
                    distance_m = self.estimate_distance(res.radius)
                    
                    self.send_landing_target(x_angle, y_angle, distance_m=distance_m)

                    self._last_send = now
                    self._stable_hits += 1
                    
                    # Log every 100th detection to QGC (to avoid spam)
                    if self._total_detections % 100 == 0 and self._total_detections > 0:
                        msg = f"PrecLand: {self._total_detections} targets sent"
                        self.send_qgc_message(msg, severity=mavutil.mavlink.MAV_SEVERITY_DEBUG)
                else:
                    self._stable_hits = 0

                # Auto-switch to LAND if desired
                # self.maybe_switch_to_land()

                time.sleep(0.01)
                
        except KeyboardInterrupt:
            print("Stopping…")
            self.send_qgc_message("PrecLand: Shutting down", severity=mavutil.mavlink.MAV_SEVERITY_WARNING)
        except Exception as e:
            print(f"Error in main loop: {e}")
            self.send_qgc_message(f"PrecLand: Error {str(e)[:30]}", severity=mavutil.mavlink.MAV_SEVERITY_ERROR)
        finally:
            self.running = False
            self.detector.cleanup()
            self.send_qgc_message("PrecLand: Stopped", severity=mavutil.mavlink.MAV_SEVERITY_INFO)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mavurl", default="serial:/dev/serial0:115200",
                        help="MAVLink URL (serial:/dev/serial0:115200, udpout:127.0.0.1:14550, etc.)")
    parser.add_argument("--hfov", type=float, default=78.0, help="Camera horizontal FOV (deg)")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--land_on_lock", action="store_true", help="Switch to LAND when target stable")
    parser.add_argument("--quiet", action="store_true", help="Disable QGC status messages")
    args = parser.parse_args()

    cam = CameraModel(width=args.width, height=args.height, hfov_deg=args.hfov)
    pub = LandingTargetPublisher(
        mavlink_url=args.mavurl, 
        camera=cam,
        verbose_qgc=not args.quiet  # Enable QGC messages unless --quiet
    )
    if args.land_on_lock:
        pub.maybe_switch_to_land = pub.maybe_switch_to_land  # already present
    pub.run()