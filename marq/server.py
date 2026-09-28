# drop-in /v1/systemone server, so the official SDK works if you point TYPESAFE_BASE_URL at it
import json, sys, threading, time, traceback
from pathlib import Path
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from .engine import BadRequest

PAGE = Path(__file__).with_name("playground.html")


def serve(engine, host="127.0.0.1", port=8765, token="", keep_warm=300):
    # the one loaded model answers everything, and jev-latest stays listed since the SDK sends it by default
    models = {"models": [{"name": n, "description": f"Local System One model {engine.name}, {engine.n_ctx}-token context.",
                          "release_date": "2026-09-26"} for n in dict.fromkeys(["jev-latest", engine.name])]}
    lock = threading.Lock()  # one engine, so requests take turns
    last = [time.time()]

    # the GPU sleeps 5 s after its last job, so keep it busy for a while after each request
    def warm():
        while True:
            time.sleep(3)
            if time.time() - last[0] < keep_warm:
                with lock:
                    engine.ping()
    if keep_warm:
        threading.Thread(target=warm, daemon=True).start()

    class H(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"
        disable_nagle_algorithm = True

        def log_message(self, *a):
            pass

        def send(self, code, obj):
            data = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def authed(self):
            if token and self.headers.get("Authorization", "") != f"Bearer {token}":
                self.send(401, {"detail": "Invalid or missing bearer token"})
                return False
            return True

        def do_GET(self):
            last[0] = time.time()
            path = self.path.split("?")[0].rstrip("/")
            if path == "/health":
                return self.send(200, {"ok": True, "model": engine.name})
            if path == "":  # the playground page, its API calls still send the token
                data = PAGE.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                return self.wfile.write(data)
            if not self.authed():
                return
            if path == "/v1/models":
                return self.send(200, models)
            self.send(404, {"detail": "Not Found"})

        def do_POST(self):
            last[0] = time.time()
            n = int(self.headers.get("Content-Length") or 0)
            raw = self.rfile.read(n)
            if not self.authed():
                return
            if self.path.split("?")[0].rstrip("/") != "/v1/systemone":
                return self.send(404, {"detail": "Not Found"})
            try:
                body = json.loads(raw or b"null")
            except ValueError as e:
                return self.send(422, {"detail": [{"loc": ["body"], "msg": f"JSON decode error: {e}", "type": "json_invalid"}]})
            t0 = time.perf_counter()
            try:
                with lock:
                    out = engine.systemone(body)
            except BadRequest as e:
                return self.send(422, {"detail": e.detail})
            except Exception:
                traceback.print_exc()
                return self.send(503, {"detail": "engine error"})
            self.send(200, out)
            print(json.dumps({"ms": round((time.perf_counter() - t0) * 1000, 1), "q": len(out["answers"]),
                              "tokens": out["usage"]["input_tokens"]}), file=sys.stderr, flush=True)

    ThreadingHTTPServer.daemon_threads = True
    httpd = ThreadingHTTPServer((host, port), H)
    print(f"marq: {engine.name} on http://{host}:{port}/v1/systemone (ctx {engine.n_ctx}), playground at http://{host}:{port}/", file=sys.stderr, flush=True)
    httpd.serve_forever()
