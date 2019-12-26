cmake_minimum_required(VERSION 3.6)
project(opengles3)
add_definitions(-D WIN32)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

#设置输出路径
SET(EXECUTABLE_OUTPUT_PATH ${opengles3_SOURCE_DIR}/bin)
#设置静态链接库输出路径
SET(LIBRARY_OUTPUT_PATH ${opengles3_SOURCE_DIR}/lib)
#设置运行时输出路径
SET(RUNTIME_OUTPUT_DIRECTORY ${opengles3_SOURCE_DIR}/bin)

set(OpenCV_DIR "C:\\opencv")

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/")

find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${opengles3_SOURCE_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${opengles3_SOURCE_DIR}/bin)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${opengles3_SOURCE_DIR}/bin)

#包含头文件目录
include_directories(${CMAKE_SOURCE_DIR}/External/Include)
#包含公共组件库目录
include_directories(${CMAKE_SOURCE_DIR}/Common/Include)

#添加子目录
add_subdirectory(Common)

set(OpenCV_LIBS opencv_core opencv_imgproc opencv_highgui opencv_imgcodecs)

#链接EGL、GLESv2 lib文件
set(TARGET_LIB
        ${CMAKE_SOURCE_DIR}/library/libGLESv2.lib
        ${CMAKE_SOURCE_DIR}/library/libEGL.lib
        ${OpenCV_LIBS}
        )

set(SOURCE_FILES Simple_Texture2D.cpp)
add_executable(opengles3 ${SOURCE_FILES})


#链接库文件
target_link_libraries(opengles3 Common ${TARGET_LIB})