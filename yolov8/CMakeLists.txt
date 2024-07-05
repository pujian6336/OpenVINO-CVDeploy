cmake_minimum_required(VERSION 3.11)
project(OpenVINO_YOLOv8 LANGUAGES CXX)

# 输出系统信息
message(STATUS "System: ${CMAKE_SYSTEM_NAME} ${CMAKE_SYSTEM_VERSION}")

# 设置编译器
set(CMAKE_CXX_STANDARD 14)

if(WIN32) # Windows 系统
    # 指定windows系统下OpenCVConfig.cmake文件搜索路径
    list(APPEND CMAKE_PREFIX_PATH E:\\lib\\opencv\\build\\x64\\vc16\\lib)
    set(OpenVINO_INCLUDEE_DIRS "E:\\lib\\openvino_2024.1.0\\runtime\\include")
    set(OpenVINO_LIBS "E:\\lib\\openvino_2024.1.0\\runtime\\lib\\intel64\\Release\\openvino.lib")
elseif(UNIX) # Linux系统
    set(OpenVINO_INCLUDEE_DIRS "/usr/include/openvino")
    set(OpenVINO_LIBS "/usr/lib/libopenvino.so")
else()
    message(FATAL_ERROR "Unsupported operating system.")
endif()

# 寻找OpenCV库
find_package(OpenCV REQUIRED)

include_directories(
    ${OpenCV_INCLUDE_DIRS}
    ${OpenVINO_INCLUDEE_DIRS}
    ${CMAKE_SOURCE_DIR}
)

message(STATUS "OpenCV library status:")
message(STATUS "version:${OpenCV_VERSION}")
message(STATUS "libraries:${OpenCV_LIBS}")
message(STATUS "include path:${OpenCV_INCLUDE_DIRS}")

set(YOLOV8_SOURCE)
list(APPEND DIRECTORY_LIST
    "engine"
    "process"
    "utils"
)

foreach(DIR ${DIRECTORY_LIST})
    file(GLOB SRC_FILES "${DIR}/*.cpp")
    list(APPEND YOLOV8_SOURCE ${SRC_FILES})
endforeach()

add_library(yolov8_shared SHARED ${YOLOV8_SOURCE})

target_include_directories(yolov8_shared
      PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include
      PUBLIC ${OpenCV_INCLUDE_DIRS}
      PUBLIC ${OpenVINO_INCLUDEE_DIRS}
)
target_link_libraries(yolov8_shared
      ${OpenCV_LIBS}
      ${OpenVINO_LIBS}
)

set_target_properties(yolov8_shared PROPERTIES OUTPUT_NAME "yolov8")

if(WIN32)
    add_library(yolov8_static STATIC ${YOLOV8_SOURCE})

    target_include_directories(yolov8_static
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include
    PUBLIC ${OpenCV_INCLUDE_DIRS}
    PUBLIC ${OpenVINO_INCLUDEE_DIRS}
    )

    target_link_libraries(yolov8_static
    ${OpenCV_LIBS}
    ${OpenVINO_LIBS}
    )

    # 设置静态库属性
    set_target_properties(yolov8_static PROPERTIES OUTPUT_NAME "yolov8")
endif()


# 设置动态库、静态库等输出目录
set_target_properties(yolov8_shared PROPERTIES LIBRARY_OUTPUT_DIRECTORY 
"${CMAKE_BINARY_DIR}/libs")

if(WIN32)
    add_executable(yolov8-det yolov8-det.cpp)
    target_link_libraries(yolov8-det yolov8_shared yolov8_static)

    add_executable(yolov8-det-save-txt yolov8_det_save_txt.cpp)
    target_link_libraries(yolov8-det-save-txt yolov8_shared yolov8_static)

    add_executable(yolov8-seg yolov8-seg.cpp)
    target_link_libraries(yolov8-seg yolov8_shared yolov8_static)

    add_executable(yolov8-pose yolov8-pose.cpp)
    target_link_libraries(yolov8-pose yolov8_shared yolov8_static)

    add_executable(yolov8-obb yolov8-obb.cpp)
    target_link_libraries(yolov8-obb yolov8_shared yolov8_static)
else()
    add_executable(yolov8-det yolov8-det.cpp)
    target_link_libraries(yolov8-det yolov8_shared)

    add_executable(yolov8-det-save-txt yolov8_det_save_txt.cpp)
    target_link_libraries(yolov8-det-save-txt yolov8_shared)

    add_executable(yolov8-seg yolov8-seg.cpp)
    target_link_libraries(yolov8-seg yolov8_shared)

    add_executable(yolov8-pose yolov8-pose.cpp)
    target_link_libraries(yolov8-pose yolov8_shared)

    add_executable(yolov8-obb yolov8-obb.cpp)
    target_link_libraries(yolov8-obb yolov8_shared)
endif()

install(TARGETS yolov8_shared
        LIBRARY DESTINATION bin # 动态库
)

install(
    DIRECTORY engine DESTINATION include
    FILES_MATCHING PATTERN "*.h"
)




