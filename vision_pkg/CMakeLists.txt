cmake_minimum_required(VERSION 3.8)
project(vision_pkg)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic -O3)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(OpenCV REQUIRED COMPONENTS core dnn imgproc highgui imgcodecs)

# Check for turbojpeg
find_path(TURBOJPEG_INCLUDE_DIR turbojpeg.h)
find_library(TURBOJPEG_LIBRARY turbojpeg)
if(TURBOJPEG_INCLUDE_DIR AND TURBOJPEG_LIBRARY)
  set(HAS_TURBOJPEG TRUE)
else()
  set(HAS_TURBOJPEG FALSE)
endif()

add_executable(vision_node src/vision_node.cpp)
target_include_directories(vision_node PRIVATE
  ${OpenCV_INCLUDE_DIRS}
  ${CMAKE_CURRENT_SOURCE_DIR}/onnxruntime/include
)
target_link_libraries(vision_node
  ${OpenCV_LIBRARIES}
  ${CMAKE_CURRENT_SOURCE_DIR}/onnxruntime/lib/libonnxruntime.so
)

if(HAS_TURBOJPEG)
  target_include_directories(vision_node PRIVATE ${TURBOJPEG_INCLUDE_DIR})
  target_link_libraries(vision_node ${TURBOJPEG_LIBRARY})
  target_compile_definitions(vision_node PRIVATE HAS_TURBOJPEG)
endif()

ament_target_dependencies(vision_node
  rclcpp
  sensor_msgs
  geometry_msgs
  cv_bridge
)

install(TARGETS
  vision_node
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY onnxruntime/lib/
  DESTINATION lib
  FILES_MATCHING PATTERN "*.so*"
)

ament_package()
