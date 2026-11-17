#!/usr/bin/env python3
"""
FloodBot actuator node (Person B).

Contract (do not rename):
  subscribes /cmd/drive   geometry_msgs/Twist
                linear.x  = throttle  -1.0 full reverse .. +1.0 full forward
                angular.z = steer     -1.0 full right   .. +1.0 full LEFT
  publishes  /drive/armed std_msgs/Bool   Pixhawk armed state (2 Hz)

Talks to the Pixhawk 4 Mini over USB with pymavlink, the same way the course
rover_node does (~/ros2-racer/src/robo_rover/robo_rover/rover_node.py):
wait for heartbeat -> set mode -> arm -> stream MANUAL_CONTROL at 20 Hz with
y = steering, z = throttle (0 = neutral, +-1000 = full).

SAFETY
  * Starts DISARMED and streams neutral. Arms only with --arm, >= 1 s after
    connecting, and only while /cmd/drive is neutral or absent. Never force-arms.
  * While disarmed it always sends neutral, whatever /cmd/drive says.
  * /cmd/drive older than --stale (default 0.3 s), or never received -> neutral.
  * Ctrl-C / SIGTERM / SIGHUP (tmux kill) / crash -> neutral x5, disarm, close.
  * If the Pixhawk is already armed at startup and --arm was not given, it is
    disarmed immediately.

Modes:
  python3 drive_node.py --probe   # no ROS: print firmware / mode / armed, exit. Never commands anything.
  python3 drive_node.py           # ROS node, stays disarmed, streams neutral
  python3 drive_node.py --arm     # ROS node, arms after 1 s if commands are neutral

Bench test (WHEELS OFF THE GROUND):
  ros2 topic pub -r 10 /cmd/drive geometry_msgs/msg/Twist "{angular: {z: 0.5}}"   # wheels must turn LEFT
  ros2 topic pub -r 10 /cmd/drive geometry_msgs/msg/Twist "{linear: {x: 0.15}}"   # wheels must spin FORWARD
  Wrong direction? restart with --steer-sign -1 / --throttle-sign +1.

Why --mode MANUAL (default) and not ACRO like rover_node: in ArduRover ACRO the
steering input is a yaw-RATE target. With the wheels raised the gyro never sees
the car turn, so the rate loop winds the steering to full lock. MANUAL passes
the steering straight through, which is what a [-1..1] steer contract means.

Note: ArduPilot only accepts MANUAL_CONTROL from the GCS whose system id equals
SYSID_MYGCS (default 255). We connect as system 255, same as rover_node.
"""

import argparse
import glob
import math
import os
import signal
import sys
import time

RATE_HZ = 20.0
ARM_DELAY_S = 1.0        # wait this long after connecting before arming
ARM_TIMEOUT_S = 5.0      # give up arming after this
HB_LOST_S = 3.0          # no Pixhawk heartbeat for this long = link lost
FORCE_DISARM_MAGIC = 21196

MAV_AUTOPILOT_ARDUPILOTMEGA = 3
AUTOPILOT_NAMES = {0: "generic", 3: "ArduPilot", 8: "invalid", 12: "PX4"}
VEHICLE_NAMES = {1: "fixed-wing", 2: "quadrotor", 10: "ground rover", 11: "boat"}


def log_print(msg):
    print("[drive_node] %s" % msg, flush=True)


def import_mavutil():
    try:
        from pymavlink import mavutil
        return mavutil
    except ImportError as e:
        raise SystemExit("[drive_node] cannot import pymavlink (%s). Fix: pip3 install --user pymavlink" % e)


# --------------------------------------------------------------------------- port

def find_port(arg):
    """'auto' = Pixhawk by USB name, else /dev/ttyACM0, else /dev/ttyACM1 (rover_node used ACM1)."""
    if arg != "auto":
        return arg
    keys = ("ardupilot", "px4", "pixhawk", "holybro", "3d_robotics", "fmu")
    hits = [p for p in sorted(glob.glob("/dev/serial/by-id/*"))
            if any(k in os.path.basename(p).lower() for k in keys)]
    hits.sort(key=lambda p: (not p.endswith("-if00"), p))  # MAVLink is on interface 00
    if hits:
        return hits[0]
    for p in ("/dev/ttyACM0", "/dev/ttyACM1"):
        if os.path.exists(p):
            return p
    raise SystemExit("[drive_node] no Pixhawk found: nothing in /dev/serial/by-id and no "
                     "/dev/ttyACM0 or /dev/ttyACM1. Is the USB cable in? Try: ls /dev/ttyACM*")


# --------------------------------------------------------------------------- MAVLink

class Pixhawk:
    """Thin pymavlink wrapper. All state comes from the receive drain (no recv_match races)."""

    def __init__(self, port, log=log_print):
        self.mavutil = import_mavutil()
        self.m = self.mavutil.mavlink
        self.port = port
        self.log = log
        self.conn = None
        self.hb = None
        self.last_hb = 0.0
        self.armed = False
        self.mode = "?"
        self.version = None
        self.acks = []

    # ---- connection
    def connect(self, baud, timeout=10.0):
        self.log("connecting to %s @ %d baud ..." % (self.port, baud))
        self.conn = self.mavutil.mavlink_connection(self.port, baud=baud, source_system=255)
        hb = self.conn.wait_heartbeat(timeout=timeout)
        if hb is None:
            raise RuntimeError("no heartbeat from %s within %.0f s. Wrong port, or another process "
                               "(rover_node, mavproxy) holding it? Check: fuser %s"
                               % (self.port, timeout, self.port))
        self._on_heartbeat(hb)
        return hb

    def close(self):
        if self.conn is not None:
            try:
                self.conn.close()
            except Exception:
                pass
            self.conn = None

    # ---- receive
    def _on_heartbeat(self, hb):
        if hb.type == self.m.MAV_TYPE_GCS or hb.autopilot == self.m.MAV_AUTOPILOT_INVALID:
            return
        if hb.get_srcSystem() != self.conn.target_system:
            return
        self.hb = hb
        self.last_hb = time.monotonic()
        self.armed = bool(hb.base_mode & self.m.MAV_MODE_FLAG_SAFETY_ARMED)
        self.mode = self.mavutil.mode_string_v10(hb)

    def drain(self, on_text=None):
        """Read everything waiting on the port. Raises if the serial device vanished."""
        while True:
            msg = self.conn.recv_msg()
            if msg is None:
                return
            t = msg.get_type()
            if t == "HEARTBEAT":
                self._on_heartbeat(msg)
            elif t == "STATUSTEXT" and on_text is not None:
                text = msg.text.rstrip("\x00").strip() if isinstance(msg.text, str) else str(msg.text)
                if text:
                    on_text(msg.severity, text)
            elif t == "COMMAND_ACK":
                self.acks.append(msg)
            elif t == "AUTOPILOT_VERSION":
                self.version = msg

    # ---- info
    @property
    def is_ardupilot(self):
        return self.hb is not None and self.hb.autopilot == MAV_AUTOPILOT_ARDUPILOTMEGA

    def describe(self):
        if self.hb is None:
            return "no heartbeat"
        return "autopilot=%s(%d) vehicle=%s(%d) mode=%s armed=%s sys=%d comp=%d" % (
            AUTOPILOT_NAMES.get(self.hb.autopilot, "?"), self.hb.autopilot,
            VEHICLE_NAMES.get(self.hb.type, "?"), self.hb.type, self.mode, self.armed,
            self.conn.target_system, self.conn.target_component)

    def version_string(self):
        if self.version is None:
            return None
        sw = self.version.flight_sw_version
        kind = {0: "dev", 64: "alpha", 128: "beta", 192: "rc", 255: "official"}.get(sw & 0xFF, str(sw & 0xFF))
        try:
            githash = bytes(self.version.flight_custom_version).decode("ascii", "ignore").strip("\x00")
        except Exception:
            githash = "?"
        return "%d.%d.%d (%s, git %s)" % ((sw >> 24) & 0xFF, (sw >> 16) & 0xFF, (sw >> 8) & 0xFF, kind, githash)

    # ---- send
    def send_control(self, steer_y, throttle_z):
        # manual_control_send(target, x, y, z, r, buttons); ArduRover: y = steering, z = throttle
        self.conn.mav.manual_control_send(self.conn.target_system, 0, int(steer_y), int(throttle_z), 0, 0)

    def send_gcs_heartbeat(self):
        self.conn.mav.heartbeat_send(self.m.MAV_TYPE_GCS, self.m.MAV_AUTOPILOT_INVALID, 0, 0, 0)

    def send_arm(self, arm, force=False):
        self.conn.mav.command_long_send(
            self.conn.target_system, self.conn.target_component,
            self.m.MAV_CMD_COMPONENT_ARM_DISARM, 0,
            1 if arm else 0, FORCE_DISARM_MAGIC if force else 0, 0, 0, 0, 0, 0)

    def request_version(self):
        c = self.conn
        c.mav.command_long_send(c.target_system, c.target_component,
                                self.m.MAV_CMD_REQUEST_MESSAGE, 0,
                                self.m.MAVLINK_MSG_ID_AUTOPILOT_VERSION, 0, 0, 0, 0, 0, 0)
        c.mav.command_long_send(c.target_system, c.target_component,
                                self.m.MAV_CMD_REQUEST_AUTOPILOT_CAPABILITIES, 0, 1, 0, 0, 0, 0, 0, 0)

    def set_mode(self, name, timeout=3.0):
        """Blocking mode change (only called at startup, while disarmed). Returns True on confirm."""
        mapping = self.conn.mode_mapping() or {}
        if name not in mapping:
            self.log("ERROR: mode %s not available. Modes: %s" % (name, ", ".join(sorted(mapping))))
            return False
        mode_id = mapping[name]
        self.conn.mav.set_mode_send(self.conn.target_system,
                                    self.m.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED, mode_id)
        t_end = time.monotonic() + timeout
        while time.monotonic() < t_end:
            self.send_control(0, 0)
            self.drain()
            if self.hb is not None and self.hb.custom_mode == mode_id:
                return True
            time.sleep(0.05)
        return False


# --------------------------------------------------------------------------- mapping

def clean(v):
    """NaN/inf -> 0, clamp to [-1, 1]."""
    try:
        v = float(v)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(v):
        return 0.0
    return max(-1.0, min(1.0, v))


def map_command(throttle, steer, p):
    """Contract values in [-1, 1] -> MANUAL_CONTROL (y, z) integers."""
    y = p.steer_sign * steer * p.steer_scale
    mag = abs(throttle)
    if mag < 0.01:
        z = 0.0
    else:
        # deadband offset gets the ESC past stiction; full scale stays at max_throttle
        z = math.copysign(p.deadband_offset + mag * (p.max_throttle - p.deadband_offset), throttle)
    z = p.throttle_sign * z
    y = max(-1000.0, min(1000.0, y))
    z = max(-float(p.max_throttle), min(float(p.max_throttle), z))
    return int(round(y)), int(round(z))


# --------------------------------------------------------------------------- shutdown

def safe_stop(px, log=log_print):
    """Neutral, disarm, close. Runs once on every exit path; ignores further signals while it runs."""
    for s in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        try:
            signal.signal(s, signal.SIG_IGN)
        except Exception:
            pass
    if px is None or px.conn is None:
        return
    log("shutdown: sending neutral and disarming ...")
    try:
        for _ in range(5):
            px.send_control(0, 0)
            time.sleep(0.05)
        for force in (False, True):
            t_cmd = time.monotonic()
            px.send_arm(False, force=force)
            t_end = t_cmd + 1.5
            while time.monotonic() < t_end:
                px.send_control(0, 0)
                px.drain()
                if px.last_hb > t_cmd and not px.armed:
                    log("shutdown: neutral sent, Pixhawk DISARMED%s." % (" (forced)" if force else ""))
                    return
                time.sleep(0.05)
            if px.last_hb <= t_cmd:
                log("shutdown: WARNING no heartbeat after disarm, cannot confirm disarmed state.")
                return
            if not force:
                log("shutdown: still armed after normal disarm, sending FORCED disarm.")
        log("shutdown: WARNING Pixhawk still reports ARMED. Kill power to the ESC!")
    except Exception as e:
        log("shutdown: error while stopping (%s). Kill power to the ESC if wheels are moving!" % e)
    finally:
        px.close()


# --------------------------------------------------------------------------- probe

def run_probe(p):
    port = find_port(p.port)
    px = Pixhawk(port)
    texts = []
    try:
        px.connect(p.baud)
        px.request_version()
        t_end = time.monotonic() + 3.0
        while time.monotonic() < t_end:
            px.drain(on_text=lambda sev, t: texts.append(t))
            if px.version is not None and time.monotonic() > t_end - 2.0:
                break
            time.sleep(0.05)
        print("port      : %s" % port)
        print("heartbeat : %s" % px.describe())
        print("version   : %s" % (px.version_string() or "no AUTOPILOT_VERSION reply"))
        for t in texts:
            print("statustext: %s" % t)
        print("modes     : %s" % ", ".join(sorted(px.conn.mode_mapping() or {})))
        if px.is_ardupilot:
            fw = "ArduRover" if px.hb.type == 10 else "ArduPilot (vehicle type %d)" % px.hb.type
            print("=> firmware: %s. drive_node supports this." % fw)
        elif px.hb is not None and px.hb.autopilot == 12:
            print("=> firmware: PX4. drive_node will NOT arm on PX4 (it is written for ArduRover).")
        else:
            print("=> firmware: unknown autopilot id %s" % (px.hb.autopilot if px.hb else "?"))
    finally:
        px.close()  # probe never sends controls or arm commands


# --------------------------------------------------------------------------- ROS node

def run_ros(p):
    import rclpy
    from rclpy.node import Node
    from geometry_msgs.msg import Twist
    from std_msgs.msg import Bool

    stop = {"flag": False}

    def on_signal(signum, frame):
        stop["flag"] = True

    # Own the signals so shutdown always runs our neutral/disarm sequence with a live context.
    try:
        from rclpy.signals import SignalHandlerOptions
        rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
    except (ImportError, TypeError):
        rclpy.init()
    for s in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(s, on_signal)

    class DriveNode(Node):
        def __init__(self, px):
            super().__init__("drive_node")
            self.px = px
            self.log = self.get_logger()
            self.cmd = (0.0, 0.0)          # (throttle, steer), cleaned
            self.cmd_t = None              # monotonic time of last /cmd/drive
            self.was_stale = True
            self.out = (0, 0)              # last (y, z) sent
            self.link_ok = True
            self.prev_armed = px.armed
            self.arm_state = "waiting" if p.arm else "off"   # waiting -> pending -> done | failed
            self.arm_deadline = 0.0

            self.log.info("Pixhawk on %s: %s" % (px.port, px.describe()))
            if not px.is_ardupilot:
                self.log.error("autopilot is not ArduPilot. Streaming neutral only; will NOT arm.")
            elif p.mode != "KEEP":
                if px.set_mode(p.mode):
                    self.log.info("mode set to %s" % p.mode)
                else:
                    self.log.error("could not confirm mode %s (current: %s)" % (p.mode, px.mode))
            if px.armed and not p.arm:
                self.log.warn("Pixhawk was already ARMED at startup and --arm not given: disarming.")
                px.send_arm(False)
            self.t_connect = time.monotonic()

            self.create_subscription(Twist, p.topic, self.on_cmd, 10)
            self.pub_armed = self.create_publisher(Bool, "/drive/armed", 10)
            self.create_timer(1.0 / RATE_HZ, self.control_tick)
            self.create_timer(0.02, self.recv_tick)
            self.create_timer(0.5, self.slow_tick)
            self.create_timer(1.0, self.log_tick)
            self.log.info("listening on %s; stale=%.2fs mode=%s arm=%s steer_sign=%+d throttle_sign=%+d "
                          "max_throttle=%d deadband=%d"
                          % (p.topic, p.stale, p.mode, p.arm, p.steer_sign, p.throttle_sign,
                             p.max_throttle, p.deadband_offset))
            if not p.arm:
                self.log.info("DISARMED (no --arm). Streaming neutral only.")

        # ---- inputs
        def on_cmd(self, msg):
            self.cmd = (clean(msg.linear.x), clean(msg.angular.z))
            self.cmd_t = time.monotonic()

        def cmd_age(self, now):
            return float("inf") if self.cmd_t is None else now - self.cmd_t

        # ---- 20 Hz output
        def control_tick(self):
            now = time.monotonic()
            stale = self.cmd_age(now) > p.stale
            if stale and not self.was_stale:
                self.log.warn("%s stale (>%.2fs): NEUTRAL" % (p.topic, p.stale))
            elif not stale and self.was_stale and self.cmd_t is not None:
                self.log.info("%s fresh again" % p.topic)
            self.was_stale = stale

            self.maybe_arm(now, stale)

            if self.px.armed and self.link_ok and not stale:
                y, z = map_command(self.cmd[0], self.cmd[1], p)
            else:
                y, z = 0, 0
            try:
                self.px.send_control(y, z)
                self.out = (y, z)
            except Exception as e:
                self.link_ok = False
                self.log.error("send failed: %s" % e, throttle_duration_sec=2.0)

        def maybe_arm(self, now, stale):
            if self.arm_state == "waiting":
                if not self.px.is_ardupilot:
                    self.arm_state = "failed"
                    self.log.error("refusing to arm: autopilot is not ArduPilot")
                    return
                if now - self.t_connect < ARM_DELAY_S or not self.link_ok:
                    return
                if not stale and (abs(self.cmd[0]) > 0.01 or abs(self.cmd[1]) > 0.01):
                    self.log.warn("waiting for a neutral %s before arming" % p.topic,
                                  throttle_duration_sec=2.0)
                    return
                self.log.info("arming ...")
                self.px.send_arm(True)
                self.arm_state = "pending"
                self.arm_deadline = now + ARM_TIMEOUT_S
            elif self.arm_state == "pending":
                if self.px.armed:
                    self.arm_state = "done"
                    self.log.warn("ARMED in %s. Bench tests: WHEELS OFF THE GROUND." % self.px.mode)
                elif now > self.arm_deadline:
                    self.arm_state = "failed"
                    self.log.error("arming FAILED (see PreArm messages above). Staying DISARMED, "
                                   "streaming neutral. Fix the cause and restart with --arm.")

        # ---- 50 Hz receive
        def on_text(self, severity, text):
            line = "[pixhawk] %s" % text
            if severity <= 3:
                self.log.error(line)
            elif severity == 4:
                self.log.warn(line)
            else:
                self.log.info(line)

        def recv_tick(self):
            try:
                self.px.drain(on_text=self.on_text)
            except Exception as e:
                self.link_ok = False
                self.log.error("receive failed (USB unplugged?): %s" % e, throttle_duration_sec=2.0)
                return
            for ack in self.px.acks:
                if ack.command == self.px.m.MAV_CMD_COMPONENT_ARM_DISARM and ack.result != 0:
                    self.log.error("arm/disarm command rejected, MAV_RESULT=%d" % ack.result)
            self.px.acks.clear()
            if self.px.armed != self.prev_armed:
                if not self.px.armed:
                    self.log.warn("Pixhawk reports DISARMED")
                self.prev_armed = self.px.armed

        # ---- 2 Hz housekeeping
        def slow_tick(self):
            now = time.monotonic()
            hb_age = now - self.px.last_hb
            if hb_age > HB_LOST_S:
                self.link_ok = False
                self.log.error("no Pixhawk heartbeat for %.1fs: link lost, commanding neutral" % hb_age,
                               throttle_duration_sec=2.0)
            elif not self.link_ok:
                self.link_ok = True
                self.log.info("Pixhawk link back")
            try:
                self.px.send_gcs_heartbeat()
            except Exception:
                pass
            self.pub_armed.publish(Bool(data=bool(self.px.armed)))

        def log_tick(self):
            now = time.monotonic()
            age = self.cmd_age(now)
            self.log.info("armed=%s mode=%s link=%s cmd_age=%s thr=%+.2f steer=%+.2f -> y=%d z=%d"
                          % (self.px.armed, self.px.mode, "ok" if self.link_ok else "LOST",
                             "never" if age == float("inf") else "%.2fs" % age,
                             self.cmd[0], self.cmd[1], self.out[0], self.out[1]))

    px = None
    node = None
    code = 0
    try:
        px = Pixhawk(find_port(p.port))
        px.connect(p.baud)
        if not stop["flag"]:
            node = DriveNode(px)
            while not stop["flag"] and rclpy.ok():
                rclpy.spin_once(node, timeout_sec=0.05)
    except SystemExit as e:
        log_print(str(e.code) if e.code not in (None, 0) else "exit")
        code = 1
    except Exception as e:
        log_print("FATAL: %s" % e)
        code = 1
    finally:
        safe_stop(px)
        if node is not None:
            node.destroy_node()
        try:
            rclpy.shutdown()
        except Exception:
            pass
    return code


# --------------------------------------------------------------------------- main

def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="FloodBot drive node: /cmd/drive -> Pixhawk (ArduRover)")
    ap.add_argument("--probe", action="store_true", help="no ROS: print firmware/mode/armed and exit")
    ap.add_argument("--port", default="auto", help="serial device, or 'auto' (default)")
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--arm", action="store_true", help="arm the Pixhawk (otherwise stays disarmed)")
    ap.add_argument("--mode", default="MANUAL", choices=["MANUAL", "ACRO", "KEEP"],
                    help="ArduRover mode to set at startup (default MANUAL; KEEP = don't change)")
    ap.add_argument("--topic", default="/cmd/drive", help="Twist topic to obey (e.g. /cmd/auto for bench tests)")
    ap.add_argument("--stale", type=float, default=0.3, help="seconds without a command before neutral")
    ap.add_argument("--steer-sign", type=int, default=1, choices=[1, -1],
                    help="flip if +angular.z turns the wheels RIGHT")
    ap.add_argument("--throttle-sign", type=int, default=-1, choices=[1, -1],
                    help="default -1: rover_node found the throttle channel inverted")
    ap.add_argument("--steer-scale", type=float, default=1000.0, help="MANUAL_CONTROL y at steer=1 (max 1000)")
    ap.add_argument("--max-throttle", type=int, default=300,
                    help="MANUAL_CONTROL z at throttle=1 (default 300 = 30%%, rover_node's clamp; max 1000)")
    ap.add_argument("--deadband-offset", type=int, default=0,
                    help="added to any non-zero throttle to beat ESC stiction (rover_node used 80)")
    p, _ = ap.parse_known_args(argv)  # ignore --ros-args etc.
    if not 0 < p.max_throttle <= 1000:
        ap.error("--max-throttle must be 1..1000")
    if not 0 <= p.deadband_offset < p.max_throttle:
        ap.error("--deadband-offset must be 0..max-throttle-1")
    if not 0 < p.steer_scale <= 1000:
        ap.error("--steer-scale must be (0, 1000]")
    if p.stale <= 0:
        ap.error("--stale must be > 0")
    return p


def main():
    p = parse_args()
    if p.probe:
        run_probe(p)
        return 0
    return run_ros(p)


if __name__ == "__main__":
    sys.exit(main())
