idf_component_register(SRCS "ha_haptic.c"
                       INCLUDE_DIRS "include"
                       REQUIRES ha_ui_core
                       PRIV_REQUIRES esp_driver_gpio esp_timer freertos)
