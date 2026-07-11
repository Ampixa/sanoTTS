"""Text -> Piper phoneme IDs for the saanoTTS S3 dashboard.

The ESP32-S3 can't run espeak, so its on-device letter-to-sound frontend garbles
anything nontrivial. This service runs on the Windows box next to the board and
does REAL espeak-ng phonemization (the same one the Kristin voice was trained
with), exposed as:  GET /ids?text=...  ->  "1,0,41,0,...,2"

The dashboard page fetches it automatically before posting to the board and
falls back to the on-device frontend if this server is down.

Run:  C:\\Users\\User\\saanotts\\venv\\Scripts\\python.exe C:\\esp\\phoneme_server.py
"""
import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

from piper.phonemize_espeak import EspeakPhonemizer
from piper.phoneme_ids import phonemes_to_ids

import os
CFG = os.environ.get("KRISTIN_ONNX_JSON", "en_US-kristin-medium.onnx.json")  # set to your voice config
PORT = 8077

try:
    with open(CFG, encoding="utf-8") as f:
        ID_MAP = json.load(f)["phoneme_id_map"]
except Exception as e:
    print("FATAL: cannot load phoneme_id_map from %s: %s" % (CFG, e))
    sys.exit(1)

PHONEMIZER = EspeakPhonemizer()


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        u = urlparse(self.path)
        if u.path != "/ids":
            self._send(404, "not found")
            return
        text = (parse_qs(u.query).get("text") or [""])[0].strip()
        if not text:
            self._send(400, "missing text")
            return
        try:
            sentences = PHONEMIZER.phonemize("en-us", text)
            flat = [p for sent in sentences for p in sent]
            ids = phonemes_to_ids(flat, ID_MAP)
            self._send(200, ",".join(str(i) for i in ids))
        except Exception as e:
            self._send(500, "error: %s" % e)

    def _send(self, code, body):
        data = body.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *args):
        pass


if __name__ == "__main__":
    print("phoneme server on 0.0.0.0:%d (espeak en-us, Kristin id map)" % PORT)
    ThreadingHTTPServer(("0.0.0.0", PORT), Handler).serve_forever()
