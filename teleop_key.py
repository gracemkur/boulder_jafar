#!/usr/bin/env python3
"""
FloodBot keyboard teleop  (owner: human-control & safety)

Run in its own terminal (needs a real TTY):
  source /opt/ros/humble/setup.bash && python3 teleop_key.py

Publishes
  /cmd/manual        geometry_msgs/Twist  at 20 Hz, continuously (feeds the watchdog)
  /cmd/mode_request  std_msgs/String      on mode keys -> control_node.py
Subscribes
  /mode              std_msgs/String      to show the current mode
(control_node.py is the only publisher of /mode and /cmd/drive.)

Keys
  W / S   throttle + / -          A / D   steer left / right  (+z = LEFT)
  SPACE   EMERGENCY STOP (latched) R       resume (stop -> manual)
  M       manual   O  auto   K  assist
  X       throttle to 0           C       center steering
  Q       quit (sends a stop request on the way out)
"""
import select
import sys
import termios
import time
import tty

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from geometry_msgs.msg import Twist
from std_msgs.msg import String

RATE_HZ = 20.0
THR_STEP = 0.05
STEER_STEP = 0.25

MODE_KEYS = {"m": "manual", "o": "auto", "k": "assist"}


def _clamp(v, lo=-1.0, hi=1.0):
    return max(lo, min(hi, v))


class Teleop(Node):
    def __init__(self):
        super().__init__("floodbot_teleop_key")
        self.pub_cmd = self.create_publisher(Twist, "/cmd/manual", 10)
        self.pub_req = self.create_publisher(String, "/cmd/mode_request", 10)
        qos_mode = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                              durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.create_subscription(String, "/mode", self._on_mode, qos_mode)
        self.thr = 0.0
        self.steer = 0.0
        self.mode = "?"
        self.last = "-"

    def _on_mode(self, m):
        self.mode = m.data

    def send_cmd(self):
        t = Twist()
        t.linear.x = float(self.thr)
        t.angular.z = float(self.steer)
        self.pub_cmd.publish(t)

    def request(self, text):
        self.pub_req.publish(String(data=text))

    def handle(self, ch):
        c = ch.lower()
        if c == " ":
            self.thr = self.steer = 0.0
            self.send_cmd()
            self.request("stop:SPACE e-stop")
            self.last = "E-STOP"
        elif c == "w":
            self.thr = _clamp(self.thr + THR_STEP)
        elif c == "s":
            self.thr = _clamp(self.thr - THR_STEP)
        elif c == "a":
            self.steer = _clamp(self.steer + STEER_STEP)   # +z = LEFT
        elif c == "d":
            self.steer = _clamp(self.steer - STEER_STEP)
        elif c == "x":
            self.thr = 0.0
        elif c == "c":
            self.steer = 0.0
        elif c == "r":
            self.thr = 0.0                  # never resume with throttle applied
            self.send_cmd()
            self.request("resume")
            self.last = "RESUME"
        elif c in MODE_KEYS:
            self.thr = 0.0                  # mode switch zeroes human throttle
            self.send_cmd()
            self.request(MODE_KEYS[c])
            self.last = MODE_KEYS[c].upper()
        else:
            return
        if c in "wsadxc":
            self.last = c.upper()

    def status(self):
        sys.stdout.write(f"\r mode={self.mode:<7} thr={self.thr:+.2f} steer={self.steer:+.2f} "
                         f"last={self.last:<8}   ")
        sys.stdout.flush()


def main():
    if not sys.stdin.isatty():
        print("teleop_key.py needs an interactive terminal (TTY).")
        sys.exit(1)
    rclpy.init()
    node = Teleop()
    print(__doc__.split("Keys", 1)[1])
    old = termios.tcgetattr(sys.stdin)
    period = 1.0 / RATE_HZ
    next_pub = time.monotonic()
    try:
        tty.setcbreak(sys.stdin.fileno())   # Ctrl-C still works
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.0)
            r, _, _ = select.select([sys.stdin], [], [], 0.01)
            if r:
                ch = sys.stdin.read(1)
                if ch.lower() == "q":
                    break
                node.handle(ch)
            now = time.monotonic()
            if now >= next_pub:
                node.send_cmd()
                node.status()
                next_pub = now + period
    except KeyboardInterrupt:
        pass
    finally:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old)
        # Clean exit: zero command + stop request. (A crash/kill -9 skips this;
        # control_node's watchdog covers that case.)
        node.thr = node.steer = 0.0
        for _ in range(3):
            node.send_cmd()
            node.request("stop:teleop exited")
            time.sleep(0.05)
        print("\nteleop exited, stop requested.")
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
