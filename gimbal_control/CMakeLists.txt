cmake_minimum_required(VERSION 3.8)
project(gimbal_control)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic -O3)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(px4_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(tf2 REQUIRED)

add_executable(gimbal_control_node src/gimbal_control_node.cpp)
ament_target_dependencies(gimbal_control_node
  rclcpp
  px4_msgs
  geometry_msgs
  tf2
)

install(TARGETS
  gimbal_control_node
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
