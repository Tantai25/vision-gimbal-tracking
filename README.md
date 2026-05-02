# ROS 2 Drone Vision & Gimbal Control (Raspberry Pi 5)

This workspace contains a high-performance system for real-time target detection (YOLO26n), low-latency streaming (FFmpeg), and automated gimbal tracking (Storm32 via PX4).

## Workspace Structure

```text
ros2_ws/
├── .gitignore
└── src/
    ├── README.md            # This file
    ├── run_system.sh        # Main startup script (Vision + Gimbal)
    ├── run_vision.sh        # Vision and Streaming only script
    ├── vision_pkg/          # Combined V4L2 Master, YOLO26n Inference, and FFmpeg Streamer
    │   └── model/           # YOLO26n model weights and ONNX file
    ├── gimbal_control/      # Vision-to-Gimbal mapping and PX4 Command Publisher
    ├── px4_msgs/            # Native PX4 ROS 2 Message definitions
    └── px4_ros_com/         # PX4-ROS 2 Communication bridge utilities
```

## How to Use

### 1. Prerequisites
Install hardware acceleration libraries:
```bash
sudo apt-get update
sudo apt-get install -y libturbojpeg0-dev ffmpeg
```

### 2. Building
```bash
cd ~/ros2_ws
colcon build --symlink-install
source install/setup.bash
```

### 3. Running
Ensure your Micro XRCE-DDS Agent is running.
```bash
./src/run_system.sh
```

### 4. Viewing Stream
On your laptop (IP: 100.100.209.42), create a file named `stream.sdp` with the content below, then run the command to receive the stream.

**stream.sdp content:**
```text
v=0
m=video 5600 RTP/AVP 96
c=IN IP4 127.0.0.1
a=rtpmap:96 H264/90000
```

**Laptop Command:**
```bash
ffplay -protocol_whitelist file,udp,rtp -i stream.sdp -flags low_delay -fflags nobuffer
```

## Gimbal Tuning

Modify parameters in `src/run_system.sh`:

| Parameter | Default | Description |
| :--- | :--- | :--- |
| gain_p | 0.02 | Pitch Speed |
| gain_y | 0.02 | Yaw Speed |
| deadzone | 0.05 | Dead-zone (5% screen) |

## Troubleshooting

- V4L2 Error: `sudo fuser -k /dev/video16`
- Inference Error: Re-export model with `opset=12`.
- Visibility: Check Tailscale and CycloneDDS config in `~/.ros/cyclonedds.xml`.
