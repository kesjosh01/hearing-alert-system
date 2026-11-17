idf_component_register(SRCS "ha_display.c"
                       INCLUDE_DIRS "include"
                       REQUIRES ha_ui_core
                       PRIV_REQUIRES esp_lcd esp_driver_spi esp_driver_gpio freertos heap)
