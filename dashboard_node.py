#!/usr/bin/env python3
"""
FloodBot web dashboard: camera + lidar feeds, wheel tests, autonomy monitor.

Run on the Pi (no extra packages: Python standard library + rclpy only):
  source /opt/ros/humble/setup.bash && python3 dashboard_node.py
  then open  http://<pi-ip>:8080  in a browser on the same network.

Test the UI without ROS or the robot (e.g. on a laptop):
  python3 dashboard_node.py --fake [--fake-image some.jpg]

Subscribes
  /perception/debug_image/compressed  sensor_msgs/CompressedImage  camera_node annotated view
  /scan                               sensor_msgs/LaserScan        RPLidar
  /perception/line_offset             std_msgs/Float32
  /perception/view_ok                 std_msgs/Bool
  /cmd/auto  /cmd/drive               geometry_msgs/Twist
  /mode                               std_msgs/String  (control_node, transient-local)
  /drive/armed                        std_msgs/Bool    (drive_node)
Publishes (exactly like teleop_key.py, so control_node stays in charge of /cmd/drive)
  /cmd/manual                         geometry_msgs/Twist  only while a browser holds control
  /cmd/mode_request                   std_msgs/String      manual|auto|assist|resume|stop:<why>
  /drive/arm_request                  std_msgs/Bool        True = arm, False = disarm (drive_node)

SAFETY
  * The dashboard never publishes /cmd/drive. control_node's watchdog, latched STOP and
    view_ok rules all still apply.
  * One driver at a time ("Take control"). Refused while another /cmd/manual publisher
    (teleop_key.py) is running; if one appears while driving, control is dropped + STOP.
  * Deadman: the driver's browser sends a command every 100 ms. Silent > 0.3 s (tab closed,
    Wi-Fi dropped) -> latched STOP request + control released. Also in AUTO.
  * Anyone can E-STOP or disarm. Mode changes and arming need control. Throttle is clamped
    to --max-throttle on the server.

HTTP API (used by dashboard.html; handy with curl)
  GET  /              dashboard.html            GET /api/state   one JSON snapshot
  GET  /events        Server-Sent Events: "state" 10 Hz, "scan" 5 Hz
  GET  /stream.mjpg   MJPEG camera stream       GET /frame.jpg   latest frame
  POST /api/take {}                  -> {ok, msg, token}
  POST /api/drive {token, thr, steer}   (this is also the deadman heartbeat)
  POST /api/release {token}   /api/mode {token, mode}   /api/arm {token, arm}   /api/estop {}
"""

import argparse
import json
import math
import os
import secrets
import signal
import socket
import threading
import time
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

HERE = os.path.dirname(os.path.abspath(__file__))
HTML_PATH = os.path.join(HERE, "dashboard.html")

DEADMAN_S = 0.3          # driver silent this long -> STOP
PUBLISH_HZ = 20.0        # /cmd/manual rate (same as teleop_key)
STATE_HZ = 10            # SSE state rate
SCAN_EVERY = 2           # send scan every N state ticks (5 Hz)
MJPEG_MAX_FPS = 15.0
SCAN_POINTS = 360

T_IMG = "/perception/debug_image/compressed"
T_SCAN = "/scan"
T_OFFSET = "/perception/line_offset"
T_VIEW = "/perception/view_ok"
T_AUTO = "/cmd/auto"
T_DRIVE = "/cmd/drive"
T_MODE = "/mode"
T_ARMED = "/drive/armed"
TOPICS = [T_IMG, T_SCAN, T_OFFSET, T_VIEW, T_AUTO, T_DRIVE, T_MODE, T_ARMED]
MODE_REQUESTS = ("manual", "auto", "assist", "resume")


def clean(v, lo=-1.0, hi=1.0):
    """NaN/inf/garbage -> 0, then clamp."""
    try:
        v = float(v)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(v):
        return 0.0
    return max(lo, min(hi, v))


class RateMeter:
    def __init__(self, window=2.0):
        self.window = window
        self.ts = deque()

    def tick(self, now):
        self.ts.append(now)
        self._trim(now)

    def _trim(self, now):
        while self.ts and now - self.ts[0] > self.window:
            self.ts.popleft()

    def hz(self, now):
        self._trim(now)
        return len(self.ts) / self.window


# --------------------------------------------------------------------------- shared state

class Hub:
    """Everything the HTTP threads and the ROS (or fake) side share. All access under self.lock."""

    def __init__(self, max_throttle, fake=False):
        self.lock = threading.Lock()
        self.frame_cond = threading.Condition(self.lock)
        self.max_throttle = max_throttle
        self.fake = fake
        self.stopping = False
        self.jpeg = None
        self.jpeg_seq = 0
        self.scan = None
        self.scan_seq = 0
        self.offset = None
        self.view_ok = None
        self.auto = None          # [thr, steer]
        self.drive = None         # [thr, steer]
        self.mode = None
        self.armed = None
        self.last = {}
        self.rates = {t: RateMeter() for t in TOPICS}
        self.driver = None        # dict(token, ip, t_hb, thr, steer)
        self.other_manual = 0     # other /cmd/manual publishers (teleop_key)
        self.mode_q = deque()
        self.arm_q = deque()
        self.zero_pending = 0
        self.events = deque(maxlen=40)

    # ---- helpers (lock held)
    def _event(self, text):
        self.events.append([time.strftime("%H:%M:%S"), text])

    def _mark(self, topic):
        now = time.monotonic()
        self.last[topic] = now
        self.rates[topic].tick(now)

    def _is_driver(self, token):
        return (self.driver is not None and isinstance(token, str)
                and secrets.compare_digest(self.driver["token"], token))

    def _drop_driver(self, reason):
        self.driver = None
        self.zero_pending = 3
        self.mode_q.append("stop:dashboard %s" % reason)
        self._event("STOP requested: %s" % reason)

    # ---- inputs from ROS / fake robot
    def event(self, text):
        with self.lock:
            self._event(text)

    def set(self, topic, attr, value):
        with self.lock:
            setattr(self, attr, value)
            self._mark(topic)

    def on_image(self, jpeg):
        with self.frame_cond:
            self.jpeg = bytes(jpeg)
            self.jpeg_seq += 1
            self._mark(T_IMG)
            self.frame_cond.notify_all()

    def on_scan(self, scan):
        with self.lock:
            self.scan = scan
            self.scan_seq += 1
            self._mark(T_SCAN)

    def get_scan(self):
        with self.lock:
            return self.scan_seq, self.scan

    def set_other_manual(self, n):
        """n = /cmd/manual publishers besides ours. If one appears mid-drive, both would feed
        control_node interleaved commands, so give up control and latch STOP."""
        with self.lock:
            self.other_manual = n
            if n > 0 and self.driver is not None:
                self._drop_driver("lost control: another /cmd/manual publisher appeared (teleop_key?)")

    # ---- browser requests -> (ok, msg)
    def take(self, ip):
        now = time.monotonic()
        with self.lock:
            d = self.driver
            if d is not None and now - d["t_hb"] <= DEADMAN_S:
                return False, "%s already has control" % d["ip"], None
            if self.other_manual > 0:
                return False, "another /cmd/manual publisher is running (teleop_key.py?). Quit it first.", None
            token = secrets.token_hex(8)
            self.driver = dict(token=token, ip=ip, t_hb=now, thr=0.0, steer=0.0)
            self._event("%s took control" % ip)
            return True, "you have control", token

    def release(self, token):
        with self.lock:
            if not self._is_driver(token):
                return False, "you do not have control"
            self._drop_driver("%s released control" % self.driver["ip"])
            return True, "released (STOP requested)"

    def drive_cmd(self, token, thr, steer):
        with self.lock:
            if not self._is_driver(token):
                return False, "you do not have control"
            cap = self.max_throttle
            self.driver.update(thr=clean(thr, -cap, cap), steer=clean(steer), t_hb=time.monotonic())
            return True, "ok"

    def request_mode(self, token, mode):
        if mode not in MODE_REQUESTS:
            return False, "unknown mode %r" % mode
        with self.lock:
            if not self._is_driver(token):
                return False, "take control first"
            self.mode_q.append(mode)
            self._event("%s requested %s" % (self.driver["ip"], mode))
            return True, "requested %s" % mode

    def estop(self, ip):
        with self.lock:
            if self.driver is not None:
                self.driver.update(thr=0.0, steer=0.0)
            self.mode_q.append("stop:dashboard E-STOP (%s)" % ip)
            self._event("E-STOP from %s" % ip)
            return True, "E-STOP sent"

    def arm(self, token, ip, arm):
        with self.lock:
            if arm:
                if not self._is_driver(token):
                    return False, "take control first"
                if abs(self.driver["thr"]) > 0.01 or abs(self.driver["steer"]) > 0.01:
                    return False, "let go of the drive keys first (command must be neutral)"
            self.arm_q.append(bool(arm))
            self._event("%s requested %s" % (ip, "ARM" if arm else "DISARM"))
            return True, "%s requested" % ("arm" if arm else "disarm")

    # ---- called at PUBLISH_HZ by the ROS/fake side
    def tick(self):
        """Run the deadman. Returns (manual_cmd or None, [mode requests], [arm requests])."""
        now = time.monotonic()
        with self.lock:
            d = self.driver
            if d is not None and now - d["t_hb"] > DEADMAN_S:
                self._drop_driver("heartbeat lost from %s (%.2fs)" % (d["ip"], now - d["t_hb"]))
                d = None
            if d is not None:
                cmd = (d["thr"], d["steer"])
            elif self.zero_pending > 0:
                self.zero_pending -= 1
                cmd = (0.0, 0.0)
            else:
                cmd = None
            modes = list(self.mode_q)
            arms = list(self.arm_q)
            self.mode_q.clear()
            self.arm_q.clear()
        return cmd, modes, arms

    def snapshot(self):
        now = time.monotonic()
        with self.lock:
            off = self.offset
            d = self.driver
            return dict(
                fake=self.fake, mode=self.mode, armed=self.armed,
                offset=off if (off is not None and math.isfinite(off)) else None,
                offset_nan=off is not None and not math.isfinite(off),
                view_ok=self.view_ok, auto=self.auto, drive=self.drive,
                ages={t: (round(now - self.last[t], 2) if t in self.last else None) for t in TOPICS},
                hz={t: round(self.rates[t].hz(now), 1) for t in TOPICS},
                cap=self.max_throttle, other_manual=self.other_manual,
                driver=None if d is None else dict(ip=d["ip"], thr=d["thr"], steer=d["steer"]),
                events=list(self.events)[-15:],
            )

    def shutdown(self):
        with self.frame_cond:
            self.stopping = True
            self.frame_cond.notify_all()


def downsample_scan(angle_min, angle_inc, ranges, range_min, range_max):
    n = len(ranges)
    step = max(1, int(math.ceil(n / float(SCAN_POINTS))))
    pts = []
    for i in range(0, n, step):
        r = ranges[i]
        if math.isfinite(r) and range_min <= r <= range_max:
            pts.append([round(angle_min + i * angle_inc, 4), round(r, 3)])
    return {"pts": pts, "range_max": round(float(range_max), 2)}


# --------------------------------------------------------------------------- HTTP

def make_handler(hub):
    stream_errors = (BrokenPipeError, ConnectionResetError, ConnectionAbortedError, OSError)

    class Handler(BaseHTTPRequestHandler):
        server_version = "FloodBotDash/1.0"

        def log_message(self, fmt, *args):  # keep the console readable
            pass

        def _send(self, code, body, ctype):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, code, obj):
            self._send(code, json.dumps(obj).encode(), "application/json")

        def do_GET(self):
            path = urlparse(self.path).path
            if path in ("/", "/index.html"):
                try:
                    with open(HTML_PATH, "rb") as f:
                        body = f.read()
                except OSError as e:
                    return self._send(500, ("dashboard.html missing: %s" % e).encode(), "text/plain")
                return self._send(200, body, "text/html; charset=utf-8")
            if path == "/api/state":
                return self._json(200, hub.snapshot())
            if path == "/events":
                return self._events()
            if path == "/stream.mjpg":
                return self._mjpeg()
            if path == "/frame.jpg":
                with hub.lock:
                    jpg = hub.jpeg
                if jpg is None:
                    return self._send(404, b"no frame yet", "text/plain")
                return self._send(200, jpg, "image/jpeg")
            self._send(404, b"not found", "text/plain")

        def _events(self):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Accel-Buffering", "no")
            self.end_headers()
            last_scan, n = -1, 0
            try:
                while not hub.stopping:
                    out = "event: state\ndata: %s\n\n" % json.dumps(hub.snapshot())
                    if n % SCAN_EVERY == 0:
                        seq, scan = hub.get_scan()
                        if scan is not None and seq != last_scan:
                            last_scan = seq
                            out += "event: scan\ndata: %s\n\n" % json.dumps(scan)
                    self.wfile.write(out.encode())
                    self.wfile.flush()
                    n += 1
                    time.sleep(1.0 / STATE_HZ)
            except stream_errors:
                pass

        def _mjpeg(self):
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            last, t_last = -1, 0.0
            try:
                while not hub.stopping:
                    with hub.frame_cond:
                        hub.frame_cond.wait_for(lambda: hub.jpeg_seq != last or hub.stopping, timeout=1.0)
                        seq, jpg = hub.jpeg_seq, hub.jpeg
                    if hub.stopping or jpg is None or seq == last:
                        continue
                    last = seq
                    wait = 1.0 / MJPEG_MAX_FPS - (time.monotonic() - t_last)
                    if wait > 0:
                        time.sleep(wait)
                    t_last = time.monotonic()
                    self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: %d\r\n\r\n" % len(jpg))
                    self.wfile.write(jpg)
                    self.wfile.write(b"\r\n")
                    self.wfile.flush()
            except stream_errors:
                pass

        def do_POST(self):
            path = urlparse(self.path).path
            try:
                n = int(self.headers.get("Content-Length") or 0)
                if n > 4096:
                    return self._json(413, {"ok": False, "msg": "body too large"})
                body = json.loads(self.rfile.read(n).decode() or "{}") if n else {}
                if not isinstance(body, dict):
                    raise ValueError("body must be a JSON object")
            except ValueError as e:
                return self._json(400, {"ok": False, "msg": "bad request: %s" % e})
            ip = self.client_address[0]
            token = body.get("token")
            if path == "/api/take":
                ok, msg, tok = hub.take(ip)
                return self._json(200, {"ok": ok, "msg": msg, "token": tok})
            if path == "/api/release":
                ok, msg = hub.release(token)
            elif path == "/api/drive":
                ok, msg = hub.drive_cmd(token, body.get("thr"), body.get("steer"))
            elif path == "/api/mode":
                ok, msg = hub.request_mode(token, str(body.get("mode", "")))
            elif path == "/api/estop":
                ok, msg = hub.estop(ip)
            elif path == "/api/arm":
                ok, msg = hub.arm(token, ip, bool(body.get("arm")))
            else:
                return self._json(404, {"ok": False, "msg": "unknown endpoint"})
            self._json(200, {"ok": ok, "msg": msg})

    return Handler


# --------------------------------------------------------------------------- ROS side

def run_ros(p, hub, stop):
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
    from geometry_msgs.msg import Twist
    from sensor_msgs.msg import CompressedImage, LaserScan
    from std_msgs.msg import Bool, Float32, String

    try:
        from rclpy.signals import SignalHandlerOptions
        rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
    except (ImportError, TypeError):
        rclpy.init()
    install_signals(stop)

    best_effort = QoSProfile(depth=5, reliability=ReliabilityPolicy.BEST_EFFORT)
    img_qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)
    mode_qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                          durability=DurabilityPolicy.TRANSIENT_LOCAL)

    class DashNode(Node):
        def __init__(self):
            super().__init__("dashboard_node")
            self.log = self.get_logger()
            sub = self.create_subscription
            sub(CompressedImage, T_IMG, lambda m: hub.on_image(m.data), img_qos)
            sub(LaserScan, T_SCAN, self.on_scan, best_effort)
            sub(Float32, T_OFFSET, lambda m: hub.set(T_OFFSET, "offset", float(m.data)), best_effort)
            sub(Bool, T_VIEW, lambda m: hub.set(T_VIEW, "view_ok", bool(m.data)), best_effort)
            sub(Twist, T_AUTO, lambda m: hub.set(T_AUTO, "auto", [m.linear.x, m.angular.z]), best_effort)
            sub(Twist, T_DRIVE, lambda m: hub.set(T_DRIVE, "drive", [m.linear.x, m.angular.z]), best_effort)
            sub(String, T_MODE, lambda m: hub.set(T_MODE, "mode", m.data), mode_qos)
            sub(Bool, T_ARMED, lambda m: hub.set(T_ARMED, "armed", bool(m.data)), best_effort)
            self.pub_manual = self.create_publisher(Twist, "/cmd/manual", 10)
            self.pub_mode = self.create_publisher(String, "/cmd/mode_request", 10)
            self.pub_arm = self.create_publisher(Bool, "/drive/arm_request", 10)
            self.create_timer(1.0 / PUBLISH_HZ, self.tick)
            self.create_timer(0.5, self.graph_tick)

        def on_scan(self, m):
            hub.on_scan(downsample_scan(m.angle_min, m.angle_increment, m.ranges, m.range_min, m.range_max))

        def graph_tick(self):
            hub.set_other_manual(max(0, self.count_publishers("/cmd/manual") - 1))

        def publish_manual(self, thr, steer):
            t = Twist()
            t.linear.x = float(thr)
            t.angular.z = float(steer)
            self.pub_manual.publish(t)

        def tick(self):
            cmd, modes, arms = hub.tick()
            if cmd is not None:
                self.publish_manual(*cmd)
            for m in modes:
                self.pub_mode.publish(String(data=m))
                self.log.info("mode request -> %s" % m)
            for a in arms:
                self.pub_arm.publish(Bool(data=a))
                self.log.warn("arm request -> %s" % a)

    node = DashNode()
    try:
        while not stop["flag"] and rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.05)
    finally:
        with hub.lock:
            had_driver = hub.driver is not None
        if had_driver:
            try:
                for _ in range(3):
                    node.publish_manual(0.0, 0.0)
                    time.sleep(0.02)
                node.pub_mode.publish(String(data="stop:dashboard exiting"))
                node.log.warn("shutdown: driver had control, sent zeros + STOP request")
            except Exception:
                pass
        node.destroy_node()
        try:
            rclpy.shutdown()
        except Exception:
            pass


# --------------------------------------------------------------------------- fake robot

class FakeRobot:
    """Imitates camera/lidar/auto_node/control_node/drive_node well enough to test the UI."""

    def __init__(self, hub, image_path):
        self.hub = hub
        self.jpeg = None
        if image_path and os.path.exists(image_path):
            with open(image_path, "rb") as f:
                self.jpeg = f.read()
        self.mode = "stop"
        self.armed = False
        self.manual = (0.0, 0.0)
        self.manual_t = -1e9

    def _stop(self, why):
        if self.mode != "stop":
            self.mode = "stop"
            self.hub.event("[fake control] STOP_LATCHED: %s" % why)

    def _request(self, req, view_ok, manual_fresh):
        req = req.split(":", 1)[0]
        if req == "stop":
            self._stop("human E-STOP / dashboard")
        elif self.mode == "stop":
            if req == "resume" and manual_fresh:
                self.mode = "manual"
                self.hub.event("[fake control] RESUME -> manual")
            else:
                self.hub.event("[fake control] REJECTED %s: STOP latched%s" % (
                    req, "" if req != "resume" else " (/cmd/manual not fresh)"))
        elif req in ("auto", "assist") and not view_ok:
            self.hub.event("[fake control] REJECTED %s: view_ok is False" % req)
        elif req in ("manual", "auto", "assist"):
            self.mode = req
            self.hub.event("[fake control] MODE_CHANGE -> %s" % req)

    @staticmethod
    def _scan(t):
        pts = []
        hx, hy = 2.5, 1.5                                   # room half-size (m)
        ox, oy = 1.0, 0.6 * math.sin(t / 2.0)               # moving post
        for i in range(SCAN_POINTS):
            a = -math.pi + i * 2 * math.pi / SCAN_POINTS
            c, s = math.cos(a), math.sin(a)
            r = min(hx / abs(c) if abs(c) > 1e-6 else 1e9, hy / abs(s) if abs(s) > 1e-6 else 1e9)
            b = ox * c + oy * s                             # ray vs circle r=0.15
            disc = b * b - (ox * ox + oy * oy - 0.15 ** 2)
            if disc >= 0 and b - math.sqrt(disc) > 0:
                r = min(r, b - math.sqrt(disc))
            pts.append([round(a, 4), round(r, 3)])
        return {"pts": pts, "range_max": 12.0}

    def run(self, stop):
        hub = self.hub
        t0 = time.monotonic()
        n = 0
        hub.event("[fake] fake robot running: no ROS, nothing real moves")
        while not stop["flag"]:
            now = time.monotonic()
            t = now - t0
            cmd, modes, arms = hub.tick()
            if cmd is not None:
                self.manual, self.manual_t = cmd, now
            manual_fresh = now - self.manual_t < 0.5
            offset = 0.5 * math.sin(t / 3.0)
            view_ok = (t % 20.0) < 16.0
            auto = [0.15, -offset] if view_ok else [0.0, 0.0]
            for m in modes:
                self._request(m, view_ok, manual_fresh)
            if self.mode in ("manual", "assist") and not manual_fresh:
                self._stop("WATCHDOG: /cmd/manual silent")
            if self.mode in ("auto", "assist") and not view_ok:
                self._stop("PERCEPTION: view_ok False while in %s" % self.mode)
            if self.mode == "manual":
                drive = list(self.manual)
            elif self.mode == "assist":
                drive = [max(-0.15, min(0.3, self.manual[0])), self.manual[1]]
            elif self.mode == "auto":
                drive = auto
            else:
                drive = [0.0, 0.0]
            for a in arms:
                self.armed = bool(a)
                hub.event("[fake drive] %s" % ("ARMED" if a else "DISARMED"))
            hub.set(T_OFFSET, "offset", offset)
            hub.set(T_VIEW, "view_ok", view_ok)
            hub.set(T_AUTO, "auto", auto)
            hub.set(T_DRIVE, "drive", drive)
            hub.set(T_MODE, "mode", self.mode)
            hub.set(T_ARMED, "armed", self.armed)
            if n % 2 == 0 and self.jpeg is not None:
                hub.on_image(self.jpeg)
            if n % 3 == 0:
                hub.on_scan(self._scan(t))
            n += 1
            time.sleep(1.0 / PUBLISH_HZ)


# --------------------------------------------------------------------------- main

def install_signals(stop):
    def on_signal(signum, frame):
        stop["flag"] = True
    for name in ("SIGINT", "SIGTERM", "SIGHUP"):
        if hasattr(signal, name):
            signal.signal(getattr(signal, name), on_signal)


def local_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("10.255.255.255", 1))   # no packet is sent
        ip = s.getsockname()[0]
        s.close()
        return ip
    except OSError:
        return socket.gethostname()


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="FloodBot web dashboard")
    ap.add_argument("--host", default="0.0.0.0", help="bind address (127.0.0.1 = only via ssh -L tunnel)")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--max-throttle", type=float, default=0.2,
                    help="server-side cap on dashboard throttle (default 0.2, max 0.5)")
    ap.add_argument("--fake", action="store_true", help="no ROS: synthetic data to test the UI")
    ap.add_argument("--fake-image", default=os.path.join(HERE, "debug.jpg"),
                    help="(with --fake) JPEG to show as the camera feed")
    p, _ = ap.parse_known_args(argv)  # ignore --ros-args etc.
    if not 0.0 < p.max_throttle <= 0.5:
        ap.error("--max-throttle must be in (0, 0.5]")
    return p


def main():
    p = parse_args()
    hub = Hub(p.max_throttle, fake=p.fake)
    try:
        server = ThreadingHTTPServer((p.host, p.port), make_handler(hub))
    except OSError as e:
        raise SystemExit("[dashboard] cannot listen on %s:%d (%s). Already running? Try --port 8081"
                         % (p.host, p.port, e))
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, daemon=True).start()
    host = local_ip() if p.host == "0.0.0.0" else p.host
    print("[dashboard] open http://%s:%d   (%s, max throttle %.2f)"
          % (host, p.port, "FAKE robot" if p.fake else "ROS", p.max_throttle), flush=True)

    stop = {"flag": False}
    try:
        if p.fake:
            install_signals(stop)
            FakeRobot(hub, p.fake_image).run(stop)
        else:
            run_ros(p, hub, stop)
    finally:
        hub.shutdown()
        server.shutdown()
        server.server_close()
        print("[dashboard] stopped", flush=True)


if __name__ == "__main__":
    main()
