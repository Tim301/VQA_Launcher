cmake_minimum_required(VERSION 2.8)

project(VQA2)
find_package(OpenCV REQUIRED)
set(INCLUDE_DIRS ${OpenCV_INCLUDE_DIRS})
set(LIBS ${OpenCV_LIBS})
include_directories(${INCLUDE_DIRS})

find_package(nlohmann_json 3.2.0 REQUIRED)

add_executable(${PROJECT_NAME} "main.cpp")
target_link_libraries(VQA2 ${LIBS})
