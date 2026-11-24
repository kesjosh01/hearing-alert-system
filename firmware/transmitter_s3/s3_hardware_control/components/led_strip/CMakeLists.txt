idf_component_register(
    SRCS "src/led_strip_api.c" "src/led_strip_rmt_dev.c" "src/led_strip_rmt_encoder.c"
    INCLUDE_DIRS "include" "interface" "src"
    REQUIRES esp_driver_rmt esp_driver_spi heap)
