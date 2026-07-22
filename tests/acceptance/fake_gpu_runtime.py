#!/usr/bin/env python3

import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


class Handler(BaseHTTPRequestHandler):
    server_version = "fake-gpu-runtime/1"

    def log_message(self, *_args):
        return

    def _authorized(self):
        expected = "Bearer " + self.server.token
        provided = self.headers.get_all("Authorization") or []
        return len(provided) == 1 and provided[0] == expected

    def _send_json(self, status, payload):
        encoded = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def do_GET(self):
        if self.path.rstrip("/") == "/health":
            if not self._authorized():
                self._send_json(401, {"error": "unauthorized"})
                return
            self._send_json(200, {"status": "ok"})
            return
        self._send_json(404, {"error": "not found"})

    def do_POST(self):
        if not self._authorized():
            self._send_json(401, {"error": "unauthorized"})
            return
        length = int(self.headers.get("Content-Length") or "0")
        payload = json.loads(self.rfile.read(length).decode("utf-8")) if length else {}
        if self.path == "/v1/embeddings":
            inputs = payload.get("input") or []
            with open(self.server.log_path, "a", encoding="utf-8") as handle:
                handle.write(json.dumps({"path": self.path, "input": inputs}) + "\n")
            data = []
            for index, text in enumerate(inputs):
                text = str(text)
                if text == "alpha beta":
                    embedding = [1.0, 0.0]
                elif text == "alpha query":
                    embedding = [1.0, 0.0]
                else:
                    embedding = [0.0, 1.0]
                data.append({"index": index, "embedding": embedding})
            self._send_json(200, {"data": data})
            return
        self._send_json(404, {"error": "not found"})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--token", required=True)
    parser.add_argument("--log-path", required=True)
    parser.add_argument("--model", default="")
    args = parser.parse_args()

    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    server.token = args.token
    server.log_path = args.log_path
    try:
        server.serve_forever()
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
