"""ESPHome external component: on-device neural text-to-speech (sanoTTS).

Voice: en_us_e12nano, 294,642 parameters, 24 kHz, hop 256 (93.75 frames/s).
That lineage is a SIBLING of the `heart-nano` voice, not the same weights; a
measurement taken with it is never a heart-nano number.

Layout note: every source file must sit FLAT in this directory. ESPHome's
component loader (esphome/loader.py, ComponentManifest.resources) only descends
into subdirectories when a manifest sets ``recursive_sources``, which external
components do not, so a ``model/`` subdirectory would be silently dropped from
the build. It also copies only the extensions in
``esphome.const.SOURCE_FILE_EXTENSIONS`` -- {.cpp, .hpp, .h, .c, .tcc, .ino} --
which is why the Xtensa PIE kernels are carried as file-scope inline assembly
inside snt_matvec_esp32s3_asm.c rather than as a .S file.
"""

from esphome import automation, pins
import esphome.codegen as cg
from esphome.components import esp32
from esphome.components.esp32.const import VARIANT_ESP32S3
import esphome.config_validation as cv
from esphome.const import CONF_ID, CONF_VOLUME

CODEOWNERS = ["@Ampixa"]
DEPENDENCIES = ["esp32"]
MULTI_CONF = False

CONF_GATE_ON_BOOT = "gate_on_boot"
CONF_LADDER_ON_BOOT = "ladder_on_boot"
CONF_ARENA_RESERVE = "arena_reserve"
CONF_LEXICON = "lexicon"
CONF_MAIN_TASK_STACK_SIZE = "main_task_stack_size"
CONF_I2S = "i2s"
CONF_BCLK_PIN = "bclk_pin"
CONF_LRCLK_PIN = "lrclk_pin"
CONF_DOUT_PIN = "dout_pin"
CONF_TEXT = "text"
CONF_LABEL = "label"
CONF_MAX_SAMPLES = "max_samples"

sanotts_ns = cg.esphome_ns.namespace("sanotts")
SanoTTS = sanotts_ns.class_("SanoTTS", cg.Component)
SayAction = sanotts_ns.class_("SayAction", automation.Action)
GateAction = sanotts_ns.class_("GateAction", automation.Action)
LadderAction = sanotts_ns.class_("LadderAction", automation.Action)
HeapAction = sanotts_ns.class_("HeapAction", automation.Action)

# "gold" is us_gold.json alone (1,498,498 B of flash); "gold+silver" adds
# us_silver.json for another 1,547,851 B. The lexicon port's own measurement is
# that gold-only produces byte-identical ids on all 52 of its Home Assistant
# announcement sentences, so gold is the default here and silver is opt-in for
# text with a wider vocabulary. See esphome/g2p/lex/README.md.
LEXICONS = {"gold": False, "gold+silver": True}

I2S_SCHEMA = cv.Schema(
    {
        cv.Required(CONF_BCLK_PIN): pins.internal_gpio_output_pin_number,
        cv.Required(CONF_LRCLK_PIN): pins.internal_gpio_output_pin_number,
        cv.Required(CONF_DOUT_PIN): pins.internal_gpio_output_pin_number,
    }
)


CONFIG_SCHEMA = cv.Schema(
    {
            cv.GenerateID(): cv.declare_id(SanoTTS),
            cv.Optional(CONF_GATE_ON_BOOT, default=True): cv.boolean,
            cv.Optional(CONF_LADDER_ON_BOOT, default=False): cv.boolean,
            # Headroom left for WiFi, lwIP and the API server while synthesis
            # holds its arena. An arena that starves them turns a clean number
            # into a hang.
            cv.Optional(CONF_ARENA_RESERVE, default="24kB"): cv.All(
                cv.validate_bytes, cv.int_range(min=4096, max=256 * 1024)
            ),
            cv.Optional(CONF_LEXICON, default="gold"): cv.one_of(*LEXICONS, lower=True),
            cv.Optional(CONF_VOLUME, default=0.8): cv.percentage,
            # The IDF default is 3,584 B, which the lexicon G2P (measured 1,257 B
            # of stack on the host) plus the synthesis call graph do not fit in.
            # Every byte here is internal SRAM and is charged to the budget this
            # component exists to measure, so it is explicit rather than hidden.
            cv.Optional(CONF_MAIN_TASK_STACK_SIZE, default=16384): cv.int_range(
                min=8192, max=65536
            ),
        cv.Optional(CONF_I2S): I2S_SCHEMA,
    }
).extend(cv.COMPONENT_SCHEMA)


async def to_code(config):
    # Checked here rather than in CONFIG_SCHEMA because the variant is only in
    # CORE.data once the esp32 component itself has been validated, and schema
    # validation order across a config file is not something to bet on.
    variant = esp32.get_esp32_variant()
    if variant != VARIANT_ESP32S3:
        raise cv.Invalid(
            f"sanotts targets the ESP32-S3; this build is for {variant}. The runtime is "
            "portable C99 and would compile elsewhere, but the PIE SIMD kernels and every "
            "measurement in BOARDS.md are ESP32-S3 -- refusing beats quietly shipping a "
            "4x slower build that looks identical."
        )

    var = cg.new_Pvariable(config[CONF_ID])
    await cg.register_component(var, config)

    cg.add(var.set_gate_on_boot(config[CONF_GATE_ON_BOOT]))
    cg.add(var.set_ladder_on_boot(config[CONF_LADDER_ON_BOOT]))
    cg.add(var.set_arena_reserve(config[CONF_ARENA_RESERVE]))
    cg.add(var.set_volume(config[CONF_VOLUME]))

    if CONF_I2S in config:
        i2s = config[CONF_I2S]
        cg.add(
            var.set_i2s_pins(
                i2s[CONF_BCLK_PIN], i2s[CONF_LRCLK_PIN], i2s[CONF_DOUT_PIN]
            )
        )

    # The ONLY math configuration the golden gate has been run under for this
    # lineage. mcu/Makefile's test-nano-fastmath target is the reference:
    # min corr 0.984762 over the 8 fixture rows, 0.987887 on the 415-frame row
    # this component gates against. Shipping without it would be shipping an
    # ungated build.
    cg.add_build_flag("-DSNT_NANO_FAST_MATH")

    if not LEXICONS[config[CONF_LEXICON]]:
        cg.add_build_flag("-DNANO_LEX_WITH_SILVER=0")

    esp32.add_idf_sdkconfig_option(
        "CONFIG_ESP_MAIN_TASK_STACK_SIZE", config[CONF_MAIN_TASK_STACK_SIZE]
    )


@automation.register_action(
    "sanotts.say",
    SayAction,
    cv.maybe_simple_value(
        {
            cv.GenerateID(CONF_ID): cv.use_id(SanoTTS),
            cv.Required(CONF_TEXT): cv.templatable(cv.string),
        },
        key=CONF_TEXT,
    ),
    synchronous=True,
)
async def sanotts_say_to_code(config, action_id, template_arg, args):
    paren = await cg.get_variable(config[CONF_ID])
    var = cg.new_Pvariable(action_id, template_arg, paren)
    template_ = await cg.templatable(config[CONF_TEXT], args, cg.std_string)
    cg.add(var.set_text(template_))
    return var


@automation.register_action(
    "sanotts.gate",
    GateAction,
    cv.Schema({cv.GenerateID(): cv.use_id(SanoTTS)}),
    synchronous=True,
)
async def sanotts_gate_to_code(config, action_id, template_arg, args):
    var = cg.new_Pvariable(action_id, template_arg)
    await cg.register_parented(var, config[CONF_ID])
    return var


@automation.register_action(
    "sanotts.ladder",
    LadderAction,
    cv.Schema({cv.GenerateID(): cv.use_id(SanoTTS)}),
    synchronous=True,
)
async def sanotts_ladder_to_code(config, action_id, template_arg, args):
    var = cg.new_Pvariable(action_id, template_arg)
    await cg.register_parented(var, config[CONF_ID])
    return var


@automation.register_action(
    "sanotts.log_heap",
    HeapAction,
    cv.maybe_simple_value(
        {
            cv.GenerateID(): cv.use_id(SanoTTS),
            cv.Required(CONF_LABEL): cv.templatable(cv.string),
        },
        key=CONF_LABEL,
    ),
    synchronous=True,
)
async def sanotts_heap_to_code(config, action_id, template_arg, args):
    var = cg.new_Pvariable(action_id, template_arg)
    await cg.register_parented(var, config[CONF_ID])
    template_ = await cg.templatable(config[CONF_LABEL], args, cg.std_string)
    cg.add(var.set_label(template_))
    return var
