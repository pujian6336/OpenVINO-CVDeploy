cmake_minimum_required(VERSION 3.11)
project(OpenVINO_YOLOv5 LANGUAGES CXX)

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

set(YOLOV5_SOURCE)
list(APPEND DIRECTORY_LIST
    "engine"
    "process"
    "utils"
)

foreach(DIR ${DIRECTORY_LIST})
    file(GLOB SRC_FILES "${DIR}/*.cpp")
    list(APPEND YOLOV5_SOURCE ${SRC_FILES})
endforeach()

add_library(yolov5_shared SHARED ${YOLOV5_SOURCE})

target_include_directories(yolov5_shared
      PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include
      PUBLIC ${OpenCV_INCLUDE_DIRS}
      PUBLIC ${OpenVINO_INCLUDEE_DIRS}
)
target_link_libraries(yolov5_shared
      ${OpenCV_LIBS}
      ${OpenVINO_LIBS}
)

set_target_properties(yolov5_shared PROPERTIES OUTPUT_NAME "yolov5")

if(WIN32)
    add_library(yolov5_static STATIC ${YOLOV5_SOURCE})

    target_include_directories(yolov5_static
      PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include
      PUBLIC ${OpenCV_INCLUDE_DIRS}
      PUBLIC ${OpenVINO_INCLUDEE_DIRS}
    )

    target_link_libraries(yolov5_static
      ${OpenCV_LIBS}
      ${OpenVINO_LIBS}
    )

    # 设置静态库属性
    set_target_properties(yolov5_static PROPERTIES OUTPUT_NAME "yolov5")
endif()


# 设置动态库、静态库等输出目录
set_target_properties(yolov5_shared PROPERTIES LIBRARY_OUTPUT_DIRECTORY 
"${CMAKE_BINARY_DIR}/libs")
if(WIN32) # Windows 系统
    add_executable(yolov5_det yolov5-det.cpp)
    target_link_libraries(yolov5_det yolov5_static yolov5_shared)

    add_executable(yolov5_det_save_txt yolov5_det_save_txt.cpp)
    target_link_libraries(yolov5_det_save_txt yolov5_static yolov5_shared)

    add_executable(yolov5_seg yolov5-seg.cpp)
    target_link_libraries(yolov5_seg yolov5_static yolov5_shared)
else()
    add_executable(yolov5_det yolov5-det.cpp)
    target_link_libraries(yolov5_det yolov5_shared)

    add_executable(yolov5_det_save_txt yolov5_det_save_txt.cpp)
    target_link_libraries(yolov5_det_save_txt yolov5_shared)

    add_executable(yolov5_seg yolov5-seg.cpp)
    target_link_libraries(yolov5_seg yolov5_shared)
endif()

install(TARGETS yolov5_shared
        LIBRARY DESTINATION bin # 动态库
)

install(
    DIRECTORY engine DESTINATION include
    FILES_MATCHING PATTERN "*.h"
)




