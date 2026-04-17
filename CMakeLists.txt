cmake_minimum_required(VERSION 3.8)
project(regnonrep)

# Default to C++17 (safe even if unused)
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclpy REQUIRED)
find_package(sensor_msgs REQUIRED)

# Install Python node
install(
  PROGRAMS
    scripts/ros_non_rep.py
    scripts/Pctools.py
    scripts/odom_to_tum.py
  DESTINATION lib/${PROJECT_NAME}
)

# Install config files
install(
  DIRECTORY config
  DESTINATION share/${PROJECT_NAME}
)

# Install launch files
install(
  DIRECTORY launch
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
