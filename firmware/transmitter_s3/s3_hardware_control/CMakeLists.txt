# The following five lines of boilerplate have to be in your project's
# CMakeLists in this exact order for cmake to work correctly
idf_component_register(
    SRCS
        "main.c"
        "dtw_classifier.c"

    INCLUDE_DIRS
        "."

    PRIV_REQUIRES
        esp_driver_i2s
        esp_driver_ledc
)

