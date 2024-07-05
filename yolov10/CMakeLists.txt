cmake_minimum_required(VERSION 3.11)
project(OpenVINO_YOLOv10 LANGUAGES CXX)

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

set(YOLOV10_SOURCE)
list(APPEND DIRECTORY_LIST
    "engine"
    "process"
    "utils"
)

foreach(DIR ${DIRECTORY_LIST})
    file(GLOB SRC_FILES "${DIR}/*.cpp")
    list(APPEND YOLOV10_SOURCE ${SRC_FILES})
endforeach()

add_library(yolov10_shared SHARED ${YOLOV10_SOURCE})

target_include_directories(yolov10_shared
      PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include
      PUBLIC ${OpenCV_INCLUDE_DIRS}
      PUBLIC ${OpenVINO_INCLUDEE_DIRS}
)
target_link_libraries(yolov10_shared
      ${OpenCV_LIBS}
      ${OpenVINO_LIBS}
)

set_target_properties(yolov10_shared PROPERTIES OUTPUT_NAME "yolov10")

if(WIN32)
    add_library(yolov10_static STATIC ${YOLOV10_SOURCE})

    target_include_directories(yolov10_static
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include
    PUBLIC ${OpenCV_INCLUDE_DIRS}
    PUBLIC ${OpenVINO_INCLUDEE_DIRS}
    )

    target_link_libraries(yolov10_static
    ${OpenCV_LIBS}
    ${OpenVINO_LIBS}
    )

    # 设置静态库属性
    set_target_properties(yolov10_static PROPERTIES OUTPUT_NAME "yolov10")
endif()


# 设置动态库、静态库等输出目录
set_target_properties(yolov10_shared PROPERTIES LIBRARY_OUTPUT_DIRECTORY 
"${CMAKE_BINARY_DIR}/libs")

if(WIN32)
    add_executable(yolov10-det yolov10-det.cpp)
    target_link_libraries(yolov10-det yolov10_shared yolov10_static)

    add_executable(yolov10-det-save-txt yolov10-det-save-txt.cpp)
    target_link_libraries(yolov10-det-save-txt yolov10_shared yolov10_static)
else()
    add_executable(yolov10-det yolov10-det.cpp)
    target_link_libraries(yolov10-det yolov10_shared)

    add_executable(yolov10-det-save-txt yolov10-det-save-txt.cpp)
    target_link_libraries(yolov10-det-save-txt yolov10_shared)
endif()

install(TARGETS yolov10_shared
        LIBRARY DESTINATION bin # 动态库
)

install(
    DIRECTORY engine DESTINATION include
    FILES_MATCHING PATTERN "*.h"
)




