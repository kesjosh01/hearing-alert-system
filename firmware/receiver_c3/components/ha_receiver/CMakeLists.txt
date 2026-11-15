idf_component_register(SRCS "ha_receiver.c"
                       INCLUDE_DIRS "include"
                       REQUIRES ha_comm freertos esp_common
                       PRIV_REQUIRES esp_wifi esp_event esp_netif esp_timer esp_pm
                                     esp_hw_support nvs_flash lwip)
