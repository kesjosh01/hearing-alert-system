# Portable protocol / queue / duplicate-suppression core shared by both boards.
# No ESP-IDF API is used in src/, so the same files are unit-tested on the host
# (see test/host/). Board glue lives in firmware/*/components/.
idf_component_register(SRCS "src/ha_proto.c" "src/ha_events.c" "src/ha_tx.c"
                            "src/ha_rx.c" "src/ha_rx_pending.c" "src/ha_time.c"
                       INCLUDE_DIRS "include")
