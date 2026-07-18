cmake_minimum_required(VERSION 3.10)
project(franka_state_bridge)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Franka REQUIRED)

add_executable(franka_state_bridge franka_state_bridge.cpp)
target_link_libraries(franka_state_bridge Franka::Franka)

