idf_component_register(SRCS "ha_sender.c"
                       INCLUDE_DIRS "include"
                       REQUIRES ha_comm esp_wifi esp_event esp_netif esp_timer
                       PRIV_REQUIRES nvs_flash lwip esp_hw_support)
