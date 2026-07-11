idf_component_register(
    SRCS "fsd_e2e.c" "esp_g2p.c" "sn_matvec_s8_esp32s3.S"
    INCLUDE_DIRS "."
    EMBED_FILES "golden/model_q8.bin" "golden/front_q8.bin" "golden/e2e_ids.bin" "golden/e2e_durs.bin" "golden/e2e_audio.bin"
    PRIV_REQUIRES esp_timer espressif__esp-nn esp_driver_ledc esp_driver_gptimer esp_wifi esp_http_server esp_netif esp_event nvs_flash espeak-ng spiffs
)
target_compile_definitions(${COMPONENT_LIB} PRIVATE FSD_FAST_MATH FSD_FROZEN_NORM)
spiffs_create_partition_image(espeak ../spiffs FLASH_IN_PROJECT)
