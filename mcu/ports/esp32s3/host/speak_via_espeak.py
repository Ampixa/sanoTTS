"""Drive the S3 board with REAL espeak phonemes: text -> phoneme server -> board.
Usage: python speak_via_espeak.py "sentence one" "sentence two" ...
"""
import sys
import time
import urllib.parse
import urllib.request

import os
PHON  = os.environ.get("PHONEMIZER_URL", "http://127.0.0.1:8077/ids")
BOARD = os.environ.get("BOARD_URL", "http://<board-ip>/api/speak")  # set BOARD_URL to your board


def ids_for(text):
    u = PHON + "?" + urllib.parse.urlencode({"text": text})
    return urllib.request.urlopen(u, timeout=15).read().decode().strip()


def speak(ids):
    import urllib.error
    body = urllib.parse.urlencode({"ids": ids}).encode()
    try:
        return urllib.request.urlopen(urllib.request.Request(BOARD, data=body), timeout=45).read().decode()
    except urllib.error.HTTPError as e:
        return "HTTP %d: %s" % (e.code, e.read().decode())


for t in sys.argv[1:]:
    try:
        ids = ids_for(t)
        n = len(ids.split(","))
        print("[%3d ids] %s" % (n, t[:50]))
        print("          board: " + speak(ids))
    except Exception as e:
        print("          ERROR: %s" % e)
    time.sleep(1)
