idf_component_register(SRCS "main.c" "microphone.c" "detector.cc" "detection.c"
                       PRIV_REQUIRES ha_sender dscnn esp_driver_i2s esp_timer)
include("${CMAKE_CURRENT_LIST_DIR}/../model/select.cmake")
ha_select_s3_model("${CMAKE_CURRENT_LIST_DIR}/../../../ml/models/pursuit_20261114/candidates/native40_r06_v1")
