#!/usr/bin/env python3
"""Simple logging proxy that forwards /v1 requests to qllmd and logs bodies.

Usage: python tools/logging_proxy.py [listen_port] [target_url]
Example: python tools/logging_proxy.py 4243 http://127.0.0.1:4242
"""
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer
import requests

TARGET = sys.argv[2] if len(sys.argv) > 2 else "http://127.0.0.1:4242"


class Handler(BaseHTTPRequestHandler):
    def _forward(self):
        url = TARGET + self.path
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length) if length else b""
        print(f"----- REQUEST {self.command} {self.path} -----")
        print(body.decode('utf-8', errors='replace'))
        resp = requests.request(self.command, url, headers={k: v for k, v in self.headers.items()}, data=body, stream=True)
        print(f"----- RESPONSE {resp.status_code} -----")
        text = resp.text
        print(text[:1000])
        self.send_response(resp.status_code)
        for k, v in resp.headers.items():
            if k.lower() == 'transfer-encoding':
                continue
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(text.encode('utf-8'))

    def do_POST(self):
        self._forward()

    def do_GET(self):
        self._forward()


if __name__ == '__main__':
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 4243
    print(f"Proxy listening on 127.0.0.1:{port}, forwarding to {TARGET}")
    server = HTTPServer(('127.0.0.1', port), Handler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.server_close()
        print('Stopped')
