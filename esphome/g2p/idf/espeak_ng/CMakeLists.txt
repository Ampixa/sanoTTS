#  espeak_ng -- ESP-IDF component: espeak-ng 1.52.0 translator + the e12nano
#  phoneme front end (nano_g2p.c).
#
#  Resolved by the IDF component manager through the `path:` dependency that
#  esphome.components.esp32.add_idf_component() writes into
#  .esphome/build/<name>/src/idf_component.yml. Being a real IDF component (and
#  not part of ESPHome's src/ glob) is what makes -std=gnu11 and per-file
#  options possible at all -- ESPHome's own CMakeLists offers neither.
#
#  LICENCE: espeak-ng is GPL-3.0-or-later. See README.md.

set(ESPEAK_SRCS
    # The 27 libespeak-ng translation units, in the same order as
    # mcu/ports/esp32s3/firmware/components/espeak-ng/CMakeLists.txt so the two
    # lists can be diffed directly.
    "libespeak-ng/common.c" "libespeak-ng/mnemonics.c" "libespeak-ng/error.c" "libespeak-ng/ieee80.c"
    "libespeak-ng/compiledata.c" "libespeak-ng/compiledict.c"
    "libespeak-ng/dictionary.c" "libespeak-ng/encoding.c" "libespeak-ng/intonation.c"
    "libespeak-ng/langopts.c" "libespeak-ng/numbers.c" "libespeak-ng/phoneme.c"
    "libespeak-ng/phonemelist.c" "libespeak-ng/readclause.c" "libespeak-ng/setlengths.c"
    "libespeak-ng/soundicon.c" "libespeak-ng/spect.c" "libespeak-ng/ssml.c"
    "libespeak-ng/synthdata.c" "libespeak-ng/synthesize.c" "libespeak-ng/tr_languages.c"
    "libespeak-ng/translate.c" "libespeak-ng/translateword.c" "libespeak-ng/voices.c"
    "libespeak-ng/wavegen.c" "libespeak-ng/speech.c" "libespeak-ng/espeak_api.c"
    # ucd-tools: the Unicode character-property tables espeak's tokenizer needs.
    "ucd-tools/case.c" "ucd-tools/categories.c" "ucd-tools/ctype.c"
    "ucd-tools/proplist.c" "ucd-tools/scripts.c" "ucd-tools/tostring.c"
)

if(NOT EXISTS "${CMAKE_CURRENT_LIST_DIR}/libespeak-ng/speech.c")
    message(FATAL_ERROR
        "espeak-ng sources are not vendored. Run:\n"
        "  ${CMAKE_CURRENT_LIST_DIR}/vendor_espeak.sh")
endif()

idf_component_register(
    SRCS "nano_g2p.c" ${ESPEAK_SRCS}
    INCLUDE_DIRS "include" "ucd-tools/include"
    PRIV_INCLUDE_DIRS "." "libespeak-ng"
    REQUIRES spiffs
)

# espeak-ng is old C -- K&R-isms and implicit int->pointer conversions that
# IDF's default -std=gnu23 rejects outright. Pin gnu11 and silence upstream's
# warnings, which are not ours to fix. Applied to the whole component: nano_g2p.c
# is clean C11 and does not care which of the two it gets.
#
# `include/compat` is deliberately NOT on the include path: it exists for hosts
# missing <endian.h> (macOS), and ESP-IDF's newlib provides one. Adding it here
# would shadow newlib headers.
target_compile_options(${COMPONENT_LIB} PRIVATE -std=gnu11 -w -DHAVE_CONFIG_H)

# Keep contractions whole instead of matching the shipped front end bit for bit.
# Measurably closer to misaki (6.09% -> 5.68% token error rate); see the comment
# block at the top of nano_g2p.c and README.md before flipping it.
# target_compile_definitions(${COMPONENT_LIB} PRIVATE NANO_G2P_SPLIT_ON_APOSTROPHE=0)
