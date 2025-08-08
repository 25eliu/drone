# Raspberry Pi DroneKit Setup Commands

1. Update & install dependencies
sudo apt update
sudo apt install python3-pip python3-dev
pip3 install dronekit pymavlink
# (Optional) install MAVProxy for MAVLink proxying
pip3 install MAVProxy

2. Configure serial port
sudo usermod -a -G dialout $USER

3. Disable serial console , enable UART with pixhawk
# edit config.txt
enable_uart=1
# delete any console='' line

4. Run MAVProxy as a MAVLink proxy
mavproxy.py --master=/dev/serial0 --baudrate 57600 --out=udp:127.0.0.1:14550

5. Run python script to connect via UDP
from dronekit import connect
vehicle = connect('udp:127.0.0.1:14550', wait_ready=True)
#drone kit code
vehicle.close()

6. Alternative: Run script via serial(No MAVProxy)
vehicle = connect('/dev/ttyAMA0, wait_ready=True, baud=57600)

