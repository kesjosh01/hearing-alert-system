# ESP32-C3 SuperMini product receiver: STA + UDP, MIN_MODEM + auto light-sleep.
# Build from the repository root:
#   idf.py -C firmware/receiver_c3 -B firmware/receiver_c3/build build
cmake_minimum_required(VERSION 3.16)

set(EXTRA_COMPONENT_DIRS "${CMAKE_CURRENT_LIST_DIR}/../common")
if(NOT DEFINED SDKCONFIG)
    set(SDKCONFIG "${CMAKE_BINARY_DIR}/sdkconfig")
endif()

include($ENV{IDF_PATH}/tools/cmake/project.cmake)
project(hearing_alert_c3)
