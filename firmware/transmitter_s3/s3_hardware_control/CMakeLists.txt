cmake_minimum_required(VERSION 3.19)

set(EXTRA_COMPONENT_DIRS
    "${CMAKE_CURRENT_LIST_DIR}/../../common/ha_comm"
    "${CMAKE_CURRENT_LIST_DIR}/components"
    "${CMAKE_CURRENT_LIST_DIR}/dscnn")
if(NOT DEFINED SDKCONFIG)
    set(SDKCONFIG "${CMAKE_BINARY_DIR}/sdkconfig")
endif()
include($ENV{IDF_PATH}/tools/cmake/project.cmake)
project(s3_hardware_control)
