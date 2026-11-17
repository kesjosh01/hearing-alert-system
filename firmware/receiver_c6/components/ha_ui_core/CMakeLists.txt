# Pure-C UI logic for the C6 receiver: presenter (alert priority), vibration
# pattern table, screens and the band rasteriser + generated fonts.
# No ESP-IDF API: unit-tested and previewed on the host (test/host/).
idf_component_register(SRCS "src/ha_presenter.c" "src/ha_screens.c" "src/ha_gfx.c" "src/ha_recovery.c"
                            "src/ha_font_data.c"
                       INCLUDE_DIRS "include"
                       REQUIRES ha_comm)
