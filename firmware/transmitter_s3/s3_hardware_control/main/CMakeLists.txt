set(model_dir "${CMAKE_CURRENT_LIST_DIR}/../model/native40_r06_v1")
idf_component_register(SRCS "main.c" "microphone.c" "detector.cc" "detection.c" "alert_led.c"
                       INCLUDE_DIRS "."
                       PRIV_INCLUDE_DIRS "${model_dir}"
                       PRIV_REQUIRES ha_sender dscnn esp_driver_gpio esp_driver_i2s esp_timer led_strip)
set(contract "${model_dir}/delivered_output_contract.json")
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${contract}")
file(READ "${contract}" model_contract)
string(JSON fire_cut GET "${model_contract}" cuts_integer_units 0)
string(JSON bell_cut GET "${model_contract}" cuts_integer_units 1)
target_compile_definitions(${COMPONENT_LIB} PRIVATE HA_FIRE_CUT=${fire_cut} HA_BELL_CUT=${bell_cut})
