#!/usr/bin/env python3
"""
FloodBot autonomy node (Person B).

Contract (do not rename):
  subscribes /perception/line_offset  std_msgs/Float32  -1 far left .. 0 center .. +1 far right, NaN = no line
             /perception/view_ok      std_msgs/Bool     True = image usable
  publishes  /cmd/auto                geometry_msgs/Twist
                linear.x  = throttle [-1..1]
                angular.z = steer    [-1..1], +z = turn LEFT

Control law (proportional):  steer = -k * offset,  throttle = fixed --throttle
  Line right of center (offset > 0) -> negative z -> turn RIGHT, toward the line.

Publishes a ZERO Twist when any of these is true:
  * view_ok is False
  * offset is NaN (no line)
  * no line_offset or view_ok message newer than --stale seconds (camera died / not started)
  * on shutdown

Usage:
  source /opt/ros/humble/setup.bash
  python3 auto_node.py [--k 1.0] [--throttle 0.15] [--stale 0.5] [--rate 20]
"""

import argparse
import math
import signal
import sys
import time


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="FloodBot line-following P controller -> /cmd/auto")
    ap.add_argument("--k", type=float, default=1.0, help="steering gain: steer = -k * offset")
    ap.add_argument("--throttle", type=float, default=0.15, help="fixed forward throttle [0..1] (default 0.15)")
    ap.add_argument("--max-steer", type=float, default=1.0, help="clamp |steer| to this (default 1.0)")
    ap.add_argument("--stale", type=float, default=0.5, help="seconds before perception counts as dead")
    ap.add_argument("--rate", type=float, default=20.0, help="publish rate in Hz")
    p, _ = ap.parse_known_args(argv)  # ignore --ros-args etc.
    if not 0.0 <= p.throttle <= 1.0:
        ap.error("--throttle must be 0..1")
    if not 0.0 < p.max_steer <= 1.0:
        ap.error("--max-steer must be (0, 1]")
    if p.stale <= 0 or p.rate <= 0:
        ap.error("--stale and --rate must be > 0")
    return p


def compute(offset, view_ok, offset_age, ok_age, p):
    """Return (throttle, steer, reason). Zero command unless everything is fresh and valid."""
    if offset_age > p.stale or ok_age > p.stale:
        return 0.0, 0.0, "perception stale"
    if not view_ok:
        return 0.0, 0.0, "view_ok=False"
    if offset is None or not math.isfinite(offset):
        return 0.0, 0.0, "no line (NaN)"
    steer = max(-p.max_steer, min(p.max_steer, -p.k * offset))
    return p.throttle, steer, "tracking"


def main():
    p = parse_args()

    import rclpy
    from rclpy.node import Node
    from geometry_msgs.msg import Twist
    from std_msgs.msg import Bool, Float32

    stop = {"flag": False}

    def on_signal(signum, frame):
        stop["flag"] = True

    # Own the signals so we can still publish a final zero with a live context.
    try:
        from rclpy.signals import SignalHandlerOptions
        rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
    except (ImportError, TypeError):
        rclpy.init()
    for s in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(s, on_signal)

    class AutoNode(Node):
        def __init__(self):
            super().__init__("auto_node")
            self.log = self.get_logger()
            self.offset = None
            self.offset_t = None
            self.view_ok = False
            self.ok_t = None
            self.last = (0.0, 0.0, "starting")
            self.prev_reason = None
            self.pub = self.create_publisher(Twist, "/cmd/auto", 10)
            self.create_subscription(Float32, "/perception/line_offset", self.on_offset, 10)
            self.create_subscription(Bool, "/perception/view_ok", self.on_ok, 10)
            self.create_timer(1.0 / p.rate, self.tick)
            self.create_timer(1.0, self.log_tick)
            self.log.info("auto_node: k=%.3f throttle=%.2f max_steer=%.2f stale=%.2fs rate=%.0fHz"
                          % (p.k, p.throttle, p.max_steer, p.stale, p.rate))

        def on_offset(self, msg):
            self.offset = float(msg.data)
            self.offset_t = time.monotonic()

        def on_ok(self, msg):
            self.view_ok = bool(msg.data)
            self.ok_t = time.monotonic()

        def publish(self, throttle, steer):
            t = Twist()
            t.linear.x = float(throttle)
            t.angular.z = float(steer)
            self.pub.publish(t)

        def tick(self):
            now = time.monotonic()
            off_age = float("inf") if self.offset_t is None else now - self.offset_t
            ok_age = float("inf") if self.ok_t is None else now - self.ok_t
            throttle, steer, reason = compute(self.offset, self.view_ok, off_age, ok_age, p)
            self.publish(throttle, steer)
            self.last = (throttle, steer, reason)
            if reason != self.prev_reason:
                (self.log.info if reason == "tracking" else self.log.warn)("state -> %s" % reason)
                self.prev_reason = reason

        def log_tick(self):
            throttle, steer, reason = self.last
            off = "none" if self.offset is None else ("nan" if math.isnan(self.offset) else "%+.3f" % self.offset)
            self.log.info("%s offset=%s view_ok=%s -> throttle=%.2f steer=%+.3f"
                          % (reason, off, self.view_ok, throttle, steer))

    node = None
    try:
        node = AutoNode()
        while not stop["flag"] and rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.05)
    finally:
        if node is not None:
            try:
                for _ in range(3):
                    node.publish(0.0, 0.0)
                    time.sleep(0.02)
                node.get_logger().info("shutdown: published zero /cmd/auto")
            except Exception:
                pass
            node.destroy_node()
        try:
            rclpy.shutdown()
        except Exception:
            pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
