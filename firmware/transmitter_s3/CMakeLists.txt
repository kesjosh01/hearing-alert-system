# ESP32-S3 (LILYGO T7 S3 v1.1) sound detector and SoftAP/UDP transmitter.
# Build from the repository root:
#   idf.py -C firmware/transmitter_s3 -B build/ml_s3 build
cmake_minimum_required(VERSION 3.19)

# Shared protocol/queue core (firmware/common/ha_comm).
set(EXTRA_COMPONENT_DIRS "${CMAKE_CURRENT_LIST_DIR}/../common" "${CMAKE_CURRENT_LIST_DIR}/dscnn")
# Keep this board's sdkconfig inside its own (git-ignored) build directory so
# the S3 and C6 projects never share or overwrite a configuration.
if(NOT DEFINED SDKCONFIG)
    set(SDKCONFIG "${CMAKE_BINARY_DIR}/sdkconfig")
endif()
include($ENV{IDF_PATH}/tools/cmake/project.cmake)
project(hearing_alert_s3)
