#!/bin/bash
# Get the directory of the script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

source /opt/ros/jazzy/setup.bash
source "${SCRIPT_DIR}/../install/setup.bash"

# Export CycloneDDS settings
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export CYCLONEDDS_URI=file://$HOME/.ros/cyclonedds.xml

# Run the vision and streaming node
ros2 run vision_pkg vision_node --ros-args \
    -p device:=/dev/video0 \
    -p width:=1280 \
    -p height:=720 \
    -p fps:=24 \
    -p pixel_format:=MJPEG \
    -p model_path:="${SCRIPT_DIR}/vision_pkg/model/yolo26n.onnx" \
    -p rtp_url:=rtp://100.100.209.42:5600
