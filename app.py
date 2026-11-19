"""Run JunctionIQ locally with Python's standard library."""
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

from traffic import CONTROLLERS, LENGTH, SCENARIOS, arrivals, benchmark, simulate, train

PAGE = Path(__file__).with_name("index.html").read_bytes()
MAX_SEED = 2147483647


def parameters(body):
    if not isinstance(body, dict):
        raise ValueError("Expected a JSON object.")
    scenario = body.get("scenario")
    seed = body.get("seed")
    interventions = body.get("interventions", [])
    if scenario not in SCENARIOS or type(seed) is not int or not 0 <= seed <= MAX_SEED:
        raise ValueError("Choose a valid scenario and random seed.")
    if not isinstance(interventions, list) or len(interventions) > 8:
        raise ValueError("A run supports up to eight traffic surges.")
    schedule = []
    for item in interventions:
        if not isinstance(item, dict):
            raise ValueError("Invalid traffic surge.")
        at, direction = item.get("at"), item.get("direction")
        if type(at) is not int or not 0 <= at < LENGTH or direction not in ("NS", "EW"):
            raise ValueError("Choose a valid surge time and direction.")
        schedule.append((at, direction))
    return scenario, seed, schedule


class Handler(BaseHTTPRequestHandler):
    def send_body(self, status, body, content_type):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def send_json(self, status, body):
        self.send_body(status, json.dumps(body, separators=(",", ":")).encode(), "application/json")

    def do_GET(self):
        if urlsplit(self.path).path == "/":
            self.send_body(200, PAGE, "text/html; charset=utf-8")
        else:
            self.send_json(404, {"error": "Page not found."})

    def do_POST(self):
        path = urlsplit(self.path).path
        if path not in ("/api/run", "/api/benchmark"):
            self.send_json(404, {"error": "Endpoint not found."})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= 4096:
                raise ValueError("Invalid request size.")
            body = json.loads(self.rfile.read(length))
            scenario, seed, schedule = parameters(body)
            if path == "/api/run":
                stream = arrivals(scenario, seed, schedule)
                result = [simulate(stream, controller, self.server.values) for controller in CONTROLLERS]
            else:
                runs = body.get("runs", 20)
                if type(runs) is not int or runs not in (10, 20, 50):
                    raise ValueError("Choose 10, 20, or 50 benchmark runs.")
                result = benchmark(scenario, seed, runs, self.server.values, schedule)
        except (ValueError, UnicodeError) as error:
            self.send_json(400, {"error": str(error)})
            return
        self.send_json(200, result)


def main():
    with ThreadingHTTPServer(("127.0.0.1", 8000), Handler) as server:
        print("Training the Q-learning controller on 2,400 episodes…", flush=True)
        server.values = train()
        print("JunctionIQ is ready at http://127.0.0.1:8000", flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            print("\nJunctionIQ stopped.")


if __name__ == "__main__":
    main()
