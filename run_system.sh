#!/bin/bash
# Get the directory of the script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

source /opt/ros/jazzy/setup.bash
source "${SCRIPT_DIR}/../install/setup.bash"

# Export CycloneDDS settings
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export CYCLONEDDS_URI=file://$HOME/.ros/cyclonedds.xml

# Run the node using the new yolo26n.onnx model
ros2 run vision_pkg vision_node --ros-args \
    -p device:=/dev/video16 \
    -p width:=640 \
    -p height:=480 \
    -p fps:=30 \
    -p pixel_format:=YUYV \
    -p model_path:="${SCRIPT_DIR}/vision_pkg/model/yolo26n.onnx" \
    -p rtp_url:=rtp://100.81.232.35:5600 &

# Start Gimbal Control Node
ros2 run gimbal_control gimbal_control_node --ros-args \
    -p gain_p:=0.02 \
    -p gain_y:=0.02 \
    -p deadzone:=0.05

# Kill background processes on exit
trap "kill 0" EXIT
