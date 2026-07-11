idf_component_register(
  SRCS
    "libespeak-ng/common.c" "libespeak-ng/mnemonics.c" "libespeak-ng/error.c" "libespeak-ng/ieee80.c"
    "libespeak-ng/compiledata.c" "libespeak-ng/compiledict.c"
    "libespeak-ng/dictionary.c" "libespeak-ng/encoding.c" "libespeak-ng/intonation.c"
    "libespeak-ng/langopts.c" "libespeak-ng/numbers.c" "libespeak-ng/phoneme.c"
    "libespeak-ng/phonemelist.c" "libespeak-ng/readclause.c" "libespeak-ng/setlengths.c"
    "libespeak-ng/soundicon.c" "libespeak-ng/spect.c" "libespeak-ng/ssml.c"
    "libespeak-ng/synthdata.c" "libespeak-ng/synthesize.c" "libespeak-ng/tr_languages.c"
    "libespeak-ng/translate.c" "libespeak-ng/translateword.c" "libespeak-ng/voices.c"
    "libespeak-ng/wavegen.c" "libespeak-ng/speech.c" "libespeak-ng/espeak_api.c"
    "ucd-tools/case.c" "ucd-tools/categories.c" "ucd-tools/ctype.c"
    "ucd-tools/proplist.c" "ucd-tools/scripts.c" "ucd-tools/tostring.c"
  INCLUDE_DIRS "include" "ucd-tools/include"
  PRIV_INCLUDE_DIRS "." "libespeak-ng"
  REQUIRES spiffs
)
# espeak-ng is old C (K&R-isms, implicit int->ptr); pin gnu11 so IDF's default
# -std=gnu23 doesn't turn its quirks into hard errors, and silence its warnings.
target_compile_options(${COMPONENT_LIB} PRIVATE -std=gnu11 -w -DHAVE_CONFIG_H)
