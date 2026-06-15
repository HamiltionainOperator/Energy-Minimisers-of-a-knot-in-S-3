#!/usr/bin/env python3
"""
Live web viewer for the S³ knot-energy flow.

Serves analysis/live_view.html and streams the trajectory the C++ binary writes
(`energy_s3 … --frames K`) to the browser over Server-Sent Events, so the knot
deforms in 3-D (rotate / zoom / pan) with the energy curve drawing live as the
optimizer runs.

Usage
-----
    # 1. run the optimizer with frame capture (writes output/<knot>/trajectory.jsonl)
    make P=2 Q=3 N=1000 ITER=3000 FRAMES=10

    # 2. in another terminal, open the live viewer on that trajectory
    python3 analysis/live_view.py output/T2_3/trajectory.jsonl

It tails a growing file, so you can start the viewer before, during, or after a
run.  Reloading the page replays from the beginning.
"""
import argparse
import os
import sys
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))
HTML_PATH = os.path.join(HERE, "live_view.html")


def make_handler(traj_path):
    class Handler(BaseHTTPRequestHandler):
        def _headers(self, ctype, **extra):
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            for k, v in extra.items():
                self.send_header(k.replace("_", "-"), v)
            self.end_headers()

        def do_GET(self):
            if self.path.split("?")[0] in ("/", "/index.html"):
                with open(HTML_PATH, "rb") as f:
                    body = f.read()
                self._headers("text/html", Content_Length=str(len(body)))
                self.wfile.write(body)
            elif self.path == "/stream":
                self._headers("text/event-stream", Cache_Control="no-cache",
                              Connection="keep-alive")
                self._tail()
            else:
                self.send_error(404)

        def _tail(self):
            """Stream complete JSONL lines as they appear; keep polling for more."""
            buf, pos = "", 0
            idle = 0
            try:
                while True:
                    if os.path.exists(traj_path):
                        with open(traj_path) as f:
                            f.seek(pos)
                            chunk = f.read()
                            pos = f.tell()
                        if chunk:
                            buf += chunk
                            *lines, buf = buf.split("\n")
                            for ln in lines:
                                ln = ln.strip()
                                if ln.endswith("}"):
                                    self.wfile.write(b"data: " + ln.encode() + b"\n\n")
                                    self.wfile.flush()
                            idle = 0
                        else:
                            idle += 1
                    # heartbeat keeps the connection alive through proxies
                    if idle and idle % 50 == 0:
                        self.wfile.write(b": keep-alive\n\n")
                        self.wfile.flush()
                    time.sleep(0.1)
            except (BrokenPipeError, ConnectionResetError):
                return

        def log_message(self, *a):
            pass

    return Handler


def main():
    ap = argparse.ArgumentParser(description="Live web viewer for the knot energy flow")
    ap.add_argument("trajectory", nargs="?",
                    help="path to trajectory.jsonl (default: most recent under output/)")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--no-open", action="store_true", help="don't auto-open the browser")
    args = ap.parse_args()

    traj = args.trajectory
    if not traj:
        # pick the most recently modified trajectory.jsonl under output/
        cands = []
        for root, _, files in os.walk("output"):
            if "trajectory.jsonl" in files:
                cands.append(os.path.join(root, "trajectory.jsonl"))
        if not cands:
            sys.exit("No trajectory given and none found under output/. "
                     "Run with --frames first, e.g. make P=2 Q=3 FRAMES=10")
        traj = max(cands, key=os.path.getmtime)

    traj = os.path.abspath(traj)
    url = f"http://localhost:{args.port}/"
    print(f"Live viewer  →  {url}")
    print(f"Streaming    →  {traj}")
    if not os.path.exists(traj):
        print("(file not there yet — it will appear once the optimizer starts)")
    srv = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(traj))
    if not args.no_open:
        webbrowser.open(url)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped.")


if __name__ == "__main__":
    main()
