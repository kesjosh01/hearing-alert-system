idf_component_register(SRCS "main.c" "microphone.c" "detector.cc" "detection.c" "alert_led.c"
                       INCLUDE_DIRS "."
                       PRIV_REQUIRES ha_sender dscnn esp_driver_gpio esp_driver_i2s esp_timer led_strip)
include("${CMAKE_CURRENT_LIST_DIR}/../../model/select.cmake")
ha_select_s3_model("${CMAKE_CURRENT_LIST_DIR}/../model/native40_r06_v1")
