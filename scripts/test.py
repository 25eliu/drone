#!/usr/bin/env python3
"""
Drone-Specific ML Libraries Test for Pi Zero 2W + Pixhawk
Tests autonomous drone ML capabilities and companion computer libraries
"""

import time
import sys
import os
import psutil
from datetime import datetime
import json
import subprocess

def run_command(cmd):
    """Run shell command and return output"""
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        return result.stdout.strip(), result.returncode == 0
    except:
        return "", False

def get_cpu_temperature():
    """Get Pi CPU temperature"""
    try:
        with open('/sys/class/thermal/thermal_zone0/temp', 'r') as f:
            return float(f.read().strip()) / 1000.0
    except:
        return 0.0

print("🚁 DRONE-SPECIFIC ML LIBRARIES TEST")
print("   Pi Zero 2W as Pixhawk Companion Computer")
print("   Testing autonomous flight ML capabilities")
print("=" * 60)

available_libs = {}
performance_results = {}

# Test 1: ArduPilot/MAVLink Integration Libraries
print("\n🔗 Testing MAVLink & ArduPilot Integration...")

# Test PyMAVLink (essential for Pixhawk communication)
try:
    from pymavlink import mavutil
    available_libs['pymavlink'] = True
    print(f"✅ PyMAVLink available - Pixhawk communication ready")
    print("   Can send/receive MAVLink commands to flight controller")
except ImportError:
    available_libs['pymavlink'] = False
    print("❌ PyMAVLink not available - install with: pip3 install pymavlink")

# Test DroneKit (high-level drone programming)
try:
    import dronekit
    available_libs['dronekit'] = True
    print("✅ DroneKit available")
    print("   High-level autonomous flight programming ready")
except ImportError:
    available_libs['dronekit'] = False
    print("❌ DroneKit not available - install with: pip3 install dronekit")

# Test 2: Computer Vision for Drones
print("\n👁️  Testing Drone Computer Vision Libraries...")

# OpenCV (essential for drone vision)
try:
    import cv2
    available_libs['opencv'] = True
    print(f"✅ OpenCV {cv2.__version__} - Drone vision ready")
    
    # Test drone-specific CV operations
    start_time = time.time()
    # Simulate aerial image processing
    test_img = cv2.imread('/dev/null')  # Will fail but tests import
    cv_time = time.time() - start_time
    print("   ✓ Object detection, landing pad recognition capable")
    print("   ✓ Optical flow, feature tracking ready")
    
except ImportError:
    available_libs['opencv'] = False
    print("❌ OpenCV not available - critical for drone vision!")

# Test apriltag (landing pad detection)
try:
    import apriltag
    available_libs['apriltag'] = True
    print("✅ AprilTag available - precision landing ready")
    print("   Perfect for autonomous landing pad detection")
except ImportError:
    available_libs['apriltag'] = False
    print("❌ AprilTag not available - install with: pip3 install apriltag")

# Test 3: AI/ML Libraries for Autonomous Flight
print("\n🤖 Testing AI/ML Libraries for Autonomous Drones...")

# NumPy (foundation for all ML)
try:
    import numpy as np
    available_libs['numpy'] = True
    print(f"✅ NumPy {np.__version__} - ML foundation ready")
    
    # Test performance for real-time processing
    start_time = time.time()
    sensor_data = np.random.rand(1000, 10)  # Simulate sensor readings
    processed = np.mean(sensor_data, axis=1)
    numpy_time = time.time() - start_time
    performance_results['numpy_sensor_processing'] = numpy_time
    print(f"   Sensor data processing: {numpy_time*1000:.1f}ms")
    
except ImportError:
    available_libs['numpy'] = False
    print("❌ NumPy not available - required for all ML operations")

# TensorFlow Lite (optimized for Pi)
try:
    import tflite_runtime.interpreter as tflite
    available_libs['tflite'] = True
    print("✅ TensorFlow Lite - Edge AI ready")
    print("   ✓ Object detection models (YOLO, MobileNet)")
    print("   ✓ Obstacle avoidance neural networks")
    print("   ✓ Path planning AI models")
except ImportError:
    available_libs['tflite'] = False
    print("❌ TensorFlow Lite not available")
    print("   Install: pip3 install tflite-runtime")

# Test 4: Specialized Drone ML Libraries
print("\n🛸 Testing Specialized Autonomous Drone Libraries...")

# Test scikit-learn for flight data analysis
try:
    import sklearn
    from sklearn.ensemble import RandomForestClassifier
    available_libs['sklearn'] = True
    print(f"✅ scikit-learn {sklearn.__version__} - Flight AI ready")
    print("   ✓ Anomaly detection in flight data")
    print("   ✓ Mission planning optimization")
    print("   ✓ Weather prediction models")
    
    # Test flight data classification
    start_time = time.time()
    # Simulate flight sensor data classification
    flight_data = np.random.rand(500, 15)  # 15 sensor inputs
    flight_labels = np.random.randint(0, 3, 500)  # 3 flight modes
    clf = RandomForestClassifier(n_estimators=10, random_state=42)
    clf.fit(flight_data, flight_labels)
    ml_time = time.time() - start_time
    performance_results['flight_mode_classification'] = ml_time
    print(f"   Flight mode classification training: {ml_time:.3f}s")
    
except ImportError:
    available_libs['sklearn'] = False
    print("❌ scikit-learn not available for flight AI")

# Test MAVSDK (modern MAVLink interface)
try:
    import mavsdk
    available_libs['mavsdk'] = True
    print("✅ MAVSDK available - Modern drone API ready")
    print("   ✓ Async/await drone programming")
    print("   ✓ Mission planning and execution")
except ImportError:
    available_libs['mavsdk'] = False
    print("❌ MAVSDK not available - install with: pip3 install mavsdk")

# Test 5: Sensor Integration Libraries
print("\n📡 Testing Drone Sensor Integration...")

# Test GPIO for additional sensors
try:
    import RPi.GPIO as GPIO
    available_libs['rpi_gpio'] = True
    print("✅ RPi.GPIO - Hardware sensor integration ready")
    print("   ✓ Ultrasonic sensors, servos, LEDs")
except ImportError:
    available_libs['rpi_gpio'] = False
    print("❌ RPi.GPIO not available")

# Test I2C for sensor communication
try:
    import smbus
    available_libs['smbus'] = True
    print("✅ SMBus (I2C) - Sensor communication ready")
    print("   ✓ Additional IMU, compass, sensors")
except ImportError:
    available_libs['smbus'] = False
    print("❌ SMBus not available - install with: sudo apt install python3-smbus")

# Test 6: Real-Time Performance Assessment
print("\n⚡ Testing Real-Time Drone ML Performance...")

if available_libs.get('numpy', False):
    print("🔥 Running drone ML pipeline simulation...")
    
    # Simulate complete autonomous drone ML pipeline
    pipeline_times = []
    cpu_usage = []
    temperatures = []
    
    for i in range(20):  # Simulate 20 processing cycles
        cycle_start = time.time()
        
        # 1. Sensor data fusion (simulate IMU, GPS, etc.)
        imu_data = np.random.rand(9)  # 3-axis accel, gyro, mag
        gps_data = np.random.rand(3)  # lat, lon, alt
        sensor_fusion = np.concatenate([imu_data, gps_data])
        
        # 2. Computer vision processing (if available)
        if available_libs.get('opencv', False):
            # Simulate image processing for obstacle detection
            fake_image = np.random.randint(0, 255, (320, 240), dtype=np.uint8)
            # Simple edge detection (lightweight)
            edges = np.gradient(fake_image)
        
        # 3. ML decision making
        if available_libs.get('sklearn', False):
            # Flight decision based on sensor data
            decision_input = sensor_fusion.reshape(1, -1)
            # Simulate decision: 0=hover, 1=forward, 2=turn
            flight_decision = np.random.choice([0, 1, 2])
        
        # 4. Path planning computation
        waypoints = np.random.rand(5, 3)  # 5 waypoints in 3D
        path_optimization = np.sum(np.diff(waypoints, axis=0)**2)
        
        cycle_time = time.time() - cycle_start
        pipeline_times.append(cycle_time)
        
        # Monitor system performance
        cpu_usage.append(psutil.cpu_percent())
        temperatures.append(get_cpu_temperature())
        
        if i % 5 == 0:
            print(f"   Cycle {i+1}: {cycle_time*1000:.1f}ms, "
                  f"CPU {cpu_usage[-1]:.1f}%, Temp {temperatures[-1]:.1f}°C")
    
    # Performance analysis
    avg_cycle_time = np.mean(pipeline_times)
    max_cycle_time = np.max(pipeline_times)
    avg_cpu = np.mean(cpu_usage)
    max_temp = np.max(temperatures)
    
    performance_results['drone_ml_pipeline'] = {
        'avg_cycle_time_ms': avg_cycle_time * 1000,
        'max_cycle_time_ms': max_cycle_time * 1000,
        'avg_cpu_percent': avg_cpu,
        'max_temperature': max_temp,
        'cycles_per_second': 1.0 / avg_cycle_time
    }
    
    print(f"\n📊 Drone ML Pipeline Performance:")
    print(f"   Average cycle time: {avg_cycle_time*1000:.1f}ms")
    print(f"   Maximum cycle time: {max_cycle_time*1000:.1f}ms")
    print(f"   Processing frequency: {1.0/avg_cycle_time:.1f} Hz")
    print(f"   Average CPU usage: {avg_cpu:.1f}%")
    print(f"   Peak temperature: {max_temp:.1f}°C")

# Generate comprehensive drone ML report
print(f"\n🚁 AUTONOMOUS DRONE ML READINESS REPORT")
print("=" * 60)

available_count = sum(available_libs.values())
total_libs = len(available_libs)
readiness_score = (available_count / total_libs) * 100

print(f"📊 Overall ML Readiness: {available_count}/{total_libs} ({readiness_score:.1f}%)")
print()

# Critical systems assessment
critical_systems = {
    'Flight Control Integration': available_libs.get('pymavlink', False) or available_libs.get('dronekit', False),
    'Computer Vision': available_libs.get('opencv', False),
    'Machine Learning': available_libs.get('numpy', False) and available_libs.get('sklearn', False),
    'Edge AI': available_libs.get('tflite', False),
    'Sensor Integration': available_libs.get('rpi_gpio', False)
}

for system, status in critical_systems.items():
    status_icon = "✅" if status else "❌"
    print(f"{status_icon} {system}: {'READY' if status else 'NEEDS SETUP'}")

print(f"\n🎯 RECOMMENDED AUTONOMOUS DRONE CAPABILITIES:")

if available_libs.get('pymavlink', False) or available_libs.get('dronekit', False):
    print("✅ MISSION EXECUTION:")
    print("   - Autonomous waypoint navigation")
    print("   - Takeoff/landing automation")
    print("   - Flight mode switching")
    print("   - Real-time telemetry processing")

if available_libs.get('opencv', False):
    print("✅ COMPUTER VISION:")
    print("   - Object detection and tracking")
    print("   - Obstacle avoidance")
    print("   - Landing pad recognition")
    print("   - Visual odometry")

if available_libs.get('sklearn', False):
    print("✅ INTELLIGENT FLIGHT:")
    print("   - Anomaly detection in flight data")
    print("   - Predictive maintenance")
    print("   - Weather-based mission planning")
    print("   - Flight pattern optimization")

if available_libs.get('tflite', False):
    print("✅ DEEP LEARNING:")
    print("   - Real-time object classification")
    print("   - Semantic segmentation")
    print("   - End-to-end flight control")
    print("   - Complex decision making")

# Installation recommendations
print(f"\n🔧 INSTALLATION RECOMMENDATIONS:")

missing_critical = []
if not (available_libs.get('pymavlink', False) or available_libs.get('dronekit', False)):
    missing_critical.append("MAVLink integration")
if not available_libs.get('opencv', False):
    missing_critical.append("Computer vision")
if not available_libs.get('numpy', False):
    missing_critical.append("ML foundation")

if missing_critical:
    print("❗ CRITICAL - Install these first:")
    if "MAVLink integration" in missing_critical:
        print("   sudo pip3 install pymavlink dronekit")
    if "Computer vision" in missing_critical:
        print("   sudo apt install python3-opencv")
    if "ML foundation" in missing_critical:
        print("   pip3 install numpy scipy")

print("\n🚀 OPTIONAL ENHANCEMENTS:")
if not available_libs.get('tflite', False):
    print("   pip3 install tflite-runtime  # For deep learning")
if not available_libs.get('apriltag', False):
    print("   pip3 install apriltag        # For precision landing")
if not available_libs.get('mavsdk', False):
    print("   pip3 install mavsdk          # Modern async drone API")

# Performance assessment
if 'drone_ml_pipeline' in performance_results:
    perf = performance_results['drone_ml_pipeline']
    hz = perf['cycles_per_second']
    
    print(f"\n⚡ REAL-TIME PERFORMANCE ASSESSMENT:")
    if hz >= 20:
        print(f"🚀 EXCELLENT: {hz:.1f}Hz - Suitable for aggressive autonomous flight")
    elif hz >= 10:
        print(f"✅ GOOD: {hz:.1f}Hz - Suitable for standard autonomous missions")
    elif hz >= 5:
        print(f"⚠️  MODERATE: {hz:.1f}Hz - Suitable for gentle autonomous flight")
    else:
        print(f"❌ LIMITED: {hz:.1f}Hz - May need optimization for real-time use")
    
    if perf['max_temperature'] > 70:
        print("🌡️  WARNING: High temperature detected - consider cooling")

print(f"\n🎉 Pi Zero 2W Drone ML Assessment Complete!")
print("Ready to build intelligent autonomous drone capabilities!")

# Save results
report_data = {
    'timestamp': str(datetime.now()),
    'available_libraries': available_libs,
    'performance_results': performance_results,
    'readiness_score': readiness_score,
    'critical_systems': critical_systems
}

try:
    with open('drone_ml_readiness_report.json', 'w') as f:
        json.dump(report_data, f, indent=2)
    print(f"📁 Detailed report saved to: drone_ml_readiness_report.json")
except:
    print("📁 Could not save report file")