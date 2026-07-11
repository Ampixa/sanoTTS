import io, sys
p = r"C:\esp\espeaktest\components\espeak-ng\libespeak-ng\speech.c"
s = io.open(p, encoding="utf-8", errors="replace").read()
old = ('\tsnprintf(path_home, sizeof(path_home), "%s", path);\n'
       '\treturn GetFileLength(path_home) == -EISDIR;\n')
new = ('\tsnprintf(path_home, sizeof(path_home), "%s", path);\n'
       '\tif (GetFileLength(path_home) == -EISDIR) return 1;\n'
       '\t/* ESP/SPIFFS: flat fs, no dir stat -- accept if a data file is readable here */\n'
       '\t{ char pb[sizeof(path_home)+16]; snprintf(pb, sizeof(pb), "%s/phontab", path_home);\n'
       '\t  FILE *pf = fopen(pb, "rb"); if (pf) { fclose(pf); return 1; } }\n'
       '\treturn 0;\n')
if new.split(chr(10))[2] in s:
    print("already patched"); sys.exit(0)
if old not in s:
    print("PATTERN NOT FOUND"); sys.exit(1)
s = s.replace(old, new, 1)
io.open(p, "w", encoding="utf-8").write(s)
print("patched speech.c check_data_path")
