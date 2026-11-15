# ESP32-S3 (LILYGO T7 S3 v1.1) product transmitter: SoftAP + UDP event sender.
# Build from the repository root:
#   idf.py -C firmware/transmitter_s3 -B firmware/transmitter_s3/build build
cmake_minimum_required(VERSION 3.16)

# Shared protocol/queue core (firmware/common/ha_comm).
set(EXTRA_COMPONENT_DIRS "${CMAKE_CURRENT_LIST_DIR}/../common")
# Keep this board's sdkconfig inside its own (git-ignored) build directory so
# the S3 and C3 projects never share or overwrite a configuration.
if(NOT DEFINED SDKCONFIG)
    set(SDKCONFIG "${CMAKE_BINARY_DIR}/sdkconfig")
endif()
# dscnn/ is intentionally NOT added: it requires esp-tflite-micro and belongs to
# the ML integration step (docs/ml/MODEL_HANDOFF.md).

include($ENV{IDF_PATH}/tools/cmake/project.cmake)
project(hearing_alert_s3)
