idf_component_register(
    SRCS "c3_main.c" "../../snt_port_esp32c3.c" "../../../../src/snt_tts.c"
    INCLUDE_DIRS "." "../../../../include" "../../../../src"
    EMBED_FILES "golden/front_q8.bin" "golden/model_q8.bin" "golden/e2e_ids.bin" "golden/e2e_durs.bin" "golden/e2e_audio.bin"
    PRIV_REQUIRES esp_timer
)
target_compile_definitions(${COMPONENT_LIB} PRIVATE FSD_FAST_MATH)

target_compile_definitions(${COMPONENT_LIB} PRIVATE SNT_PROF SNT_INT_FFT SNT_ACT_LUT SNT_INT_CHAIN)
