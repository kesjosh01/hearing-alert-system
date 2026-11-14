"""Serve only the local annotation pack and accept a human review on loopback."""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dscnn_baseline.review import accept


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--port", type=int, default=8874)
    args = parser.parse_args()
    directory = args.directory.resolve()
    annotation_path = ROOT / "annotations/reviewed_events_20261114.csv"

    class Handler(SimpleHTTPRequestHandler):
        def __init__(self, *values, **kwargs):
            super().__init__(*values, directory=str(directory), **kwargs)

        def do_POST(self):
            expected = f"http://127.0.0.1:{args.port}"
            if self.path != "/api/review" or self.headers.get("Origin") != expected:
                self.send_error(403)
                return
            try:
                size = int(self.headers.get("Content-Length", "0"))
                if not 0 < size < 1024 * 1024 or self.headers.get_content_type() != "application/json":
                    raise ValueError("Expected a bounded JSON review")
                decisions = json.loads(self.rfile.read(size))
                stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
                submitted = directory / f"human_decisions_{stamp}.json"
                with submitted.open("x", encoding="utf-8") as handle:
                    json.dump(decisions, handle, ensure_ascii=False, indent=2)
                accept(submitted, directory / "inventory.json", annotation_path)
                result, code = {"accepted": True, "annotations": str(annotation_path)}, 200
            except Exception as error:
                result, code = {"accepted": False, "error": f"{type(error).__name__}: {error}"}, 400
            body = json.dumps(result, ensure_ascii=False).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    (directory / "server_pid.txt").write_text(str(os.getpid()) + "\n")
    print(f"LOCAL_REVIEW_READY http://127.0.0.1:{args.port} pid={os.getpid()}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
