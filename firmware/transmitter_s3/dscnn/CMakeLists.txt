# Integration component only. Does not fetch/install dependencies or flash a board.
# The enclosing ESP-IDF project must already provide the official component.
idf_component_register(SRCS "log_mel_frontend.cc" "dscnn_runner.cc"
                       INCLUDE_DIRS "include"
                       REQUIRES esp-tflite-micro)
