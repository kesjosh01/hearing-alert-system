idf_component_register(SRCS "ha_ui.c"
                       INCLUDE_DIRS "include"
                       PRIV_REQUIRES ha_receiver ha_display ha_haptic ha_ui_core ha_comm
                                     esp_driver_gpio esp_adc esp_timer esp_hw_support freertos)
