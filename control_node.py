#!/usr/bin/env python3
"""
FloodBot control arbiter  (owner: human-control & safety)

Run:  source /opt/ros/humble/setup.bash && python3 control_node.py

Subscribes
  /cmd/manual          geometry_msgs/Twist  human command (teleop_key.py)
  /cmd/auto            geometry_msgs/Twist  autonomy command (line follower)
  /perception/view_ok  std_msgs/Bool        perception health
  /cmd/mode_request    std_msgs/String      mode keys from teleop_key.py
                                            (manual|auto|assist|stop[:note]|resume)
Publishes (this node is the ONLY publisher of both)
  /cmd/drive           geometry_msgs/Twist  final command to the car
  /mode                std_msgs/String      manual|auto|assist|stop  (latched/transient-local)

Twist convention: linear.x = throttle [-1, 1], angular.z = steer [-1, 1], +z = LEFT.

Safety rules enforced every control cycle:
  * Watchdog: active command source silent > CMD_TIMEOUT_S  -> latched STOP
  * view_ok False / missing / stale while in auto or assist -> latched STOP
  * Non-finite (NaN/inf) command from the active source     -> latched STOP
  * STOP latches: only an explicit human resume (R) leaves it, and resume
    always lands in MANUAL (a human is in control right after a stop).
  * Node starts in STOP.
Every mode change, refusal and a 1 Hz status line are logged with timestamps
to the console and to a CSV file (for the G3 "who controls the car and why").
"""
import csv
import math
import os
import time
from datetime import datetime

# ------------------------------- tunables -----------------------------------
CONTROL_HZ = 50.0            # /cmd/drive publish rate
CMD_TIMEOUT_S = 0.5          # watchdog on the active command source
VIEW_TIMEOUT_S = 1.0         # view_ok older than this is treated as False
ASSIST_MAX_FWD = 0.30        # assist throttle cap (forward)
ASSIST_MAX_REV = 0.15        # assist throttle cap (reverse)
ASSIST_TURN_SLOWDOWN = 0.5   # at full steer the forward cap shrinks by 50 %
ASSIST_ACCEL_PER_S = 0.6     # assist: max throttle increase per second (slew)
STATUS_PERIOD_S = 1.0        # status line / CSV row period
MODE_HEARTBEAT_S = 0.5       # /mode is re-published at this period too
LOG_DIR = os.path.expanduser(os.environ.get("FLOODBOT_LOG_DIR", "~/robo-ctl/logs"))

MANUAL, AUTO, ASSIST, STOP = "manual", "auto", "assist", "stop"
SOURCE_FOR_MODE = {MANUAL: "/cmd/manual", AUTO: "/cmd/auto", ASSIST: "/cmd/manual", STOP: "none"}
CONTROLLER_FOR_MODE = {
    MANUAL: "HUMAN (full control)",
    AUTO: "AUTONOMY (/cmd/auto)",
    ASSIST: "SHARED (human steer, code-limited throttle)",
    STOP: "NOBODY (latched stop, waiting for human R)",
}


def _clamp(v, lo, hi):
    return max(lo, min(hi, v))


def _finite(*vals):
    return all(math.isfinite(v) for v in vals)


class Arbiter:
    """Pure decision logic, no ROS imports, so it can be unit-tested anywhere.

    All times are monotonic seconds supplied by the caller.
    log_fn(event, from_mode, to_mode, reason) is called on every change/refusal.
    """

    def __init__(self, log_fn):
        self.log = log_fn
        self.mode = STOP
        self.stop_reason = "startup (latched until human presses R)"
        self.manual = None   # (throttle, steer, t)
        self.auto = None     # (throttle, steer, t)
        self.view = None     # (ok, t)
        self._assist_thr = 0.0
        self._last_step = None

    # ---- inputs ------------------------------------------------------------
    def on_manual(self, thr, steer, now):
        self.manual = (thr, steer, now)

    def on_auto(self, thr, steer, now):
        self.auto = (thr, steer, now)

    def on_view(self, ok, now):
        self.view = (bool(ok), now)

    # ---- helpers -----------------------------------------------------------
    @staticmethod
    def _age(sample, now):
        return math.inf if sample is None else now - sample[-1]

    def _active_sample(self):
        return self.auto if self.mode == AUTO else self.manual

    def view_good(self, now):
        return self.view is not None and self.view[0] and self._age(self.view, now) <= VIEW_TIMEOUT_S

    def view_desc(self, now):
        if self.view is None:
            return "missing (never received)"
        age = self._age(self.view, now)
        if age > VIEW_TIMEOUT_S:
            return f"stale ({age:.2f}s old)"
        return "True" if self.view[0] else "False"

    def _set_mode(self, new, reason, event="MODE_CHANGE"):
        old = self.mode
        if new == old:
            return
        self.mode = new
        if new == STOP:
            self.stop_reason = reason
        self._assist_thr = 0.0
        self.log(event, old, new, reason)

    def latch_stop(self, reason):
        self._set_mode(STOP, reason, "STOP_LATCHED")

    def _can_enter(self, mode, now):
        src = self.auto if mode == AUTO else self.manual
        age = self._age(src, now)
        if age > CMD_TIMEOUT_S:
            return False, f"{SOURCE_FOR_MODE[mode]} not fresh (age {age:.2f}s)"
        if mode in (AUTO, ASSIST) and not self.view_good(now):
            return False, f"view_ok is {self.view_desc(now)}"
        return True, ""

    # ---- human requests ----------------------------------------------------
    def request(self, raw, now):
        req, _, note = raw.strip().partition(":")
        req = req.strip().lower()

        if req == STOP:
            if self.mode != STOP:
                self.latch_stop(f"human E-STOP ({note.strip() or 'SPACE'})")
            return

        if self.mode == STOP:
            if req == "resume":
                ok, why = self._can_enter(MANUAL, now)
                if not ok:
                    self.log("REJECTED", STOP, MANUAL, f"resume refused: {why}")
                    return
                self._set_mode(MANUAL, "human resume (R) -> manual", "RESUME")
            else:
                self.log("REJECTED", STOP, req, f"'{req}' ignored: STOP is latched, press R first")
            return

        if req == "resume":
            return  # not stopped: nothing to resume
        if req not in (MANUAL, AUTO, ASSIST):
            self.log("REJECTED", self.mode, req, "unknown mode request")
            return
        ok, why = self._can_enter(req, now)
        if not ok:
            self.log("REJECTED", self.mode, req, why)
            return
        self._set_mode(req, f"human request ({req})")

    # ---- one control cycle -------------------------------------------------
    def step(self, now):
        """Run safety checks and return the (throttle, steer) to publish."""
        dt = 0.0 if self._last_step is None else max(0.0, now - self._last_step)
        self._last_step = now

        if self.mode != STOP:
            age = self._age(self._active_sample(), now)
            if age > CMD_TIMEOUT_S:
                self.latch_stop(f"WATCHDOG: {SOURCE_FOR_MODE[self.mode]} silent "
                                f"{age:.2f}s > {CMD_TIMEOUT_S}s (lost commands)")
            elif self.mode in (AUTO, ASSIST) and not self.view_good(now):
                self.latch_stop(f"PERCEPTION: view_ok {self.view_desc(now)} while in {self.mode}")

        if self.mode == STOP:
            return 0.0, 0.0

        thr, steer = self._active_sample()[:2]
        if not _finite(thr, steer):
            self.latch_stop(f"non-finite command on {SOURCE_FOR_MODE[self.mode]}")
            return 0.0, 0.0
        steer = _clamp(steer, -1.0, 1.0)

        if self.mode == ASSIST:
            fwd_cap = ASSIST_MAX_FWD * (1.0 - ASSIST_TURN_SLOWDOWN * abs(steer))
            target = _clamp(thr, -ASSIST_MAX_REV, fwd_cap)
            cur = self._assist_thr
            if target * cur < 0:          # changing direction: brake to 0 first
                cur = 0.0
            if abs(target) <= abs(cur):   # slowing down is always immediate
                cur = target
            else:                         # speeding up is rate-limited
                step = ASSIST_ACCEL_PER_S * dt
                cur += _clamp(target - cur, -step, step)
            self._assist_thr = cur
            thr = cur

        return _clamp(thr, -1.0, 1.0), steer


# ================================ ROS node ===================================
def main():
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
    from geometry_msgs.msg import Twist
    from std_msgs.msg import String, Bool

    # BEST_EFFORT subscriptions match both reliable and best-effort publishers.
    qos_in = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT,
                        history=HistoryPolicy.KEEP_LAST)
    qos_reliable = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE)
    qos_mode = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                          durability=DurabilityPolicy.TRANSIENT_LOCAL)

    class ControlNode(Node):
        def __init__(self):
            super().__init__("floodbot_control")
            os.makedirs(LOG_DIR, exist_ok=True)
            path = os.path.join(LOG_DIR, datetime.now().strftime("control_%Y%m%d_%H%M%S.csv"))
            self._csv_f = open(path, "w", newline="")
            self._csv = csv.writer(self._csv_f)
            self._csv.writerow(["wall_time", "mono_s", "event", "from_mode", "to_mode",
                                "in_control", "reason", "throttle_out", "steer_out"])
            self._t0 = time.monotonic()
            self._out = (0.0, 0.0)

            self.arb = Arbiter(self._log_event)

            self.pub_drive = self.create_publisher(Twist, "/cmd/drive", qos_reliable)
            self.pub_mode = self.create_publisher(String, "/mode", qos_mode)

            self.create_subscription(Twist, "/cmd/manual", self._on_manual, qos_in)
            self.create_subscription(Twist, "/cmd/auto", self._on_auto, qos_in)
            self.create_subscription(Bool, "/perception/view_ok", self._on_view, qos_in)
            self.create_subscription(String, "/cmd/mode_request", self._on_request, qos_reliable)

            self.create_timer(1.0 / CONTROL_HZ, self._control_tick)
            self.create_timer(MODE_HEARTBEAT_S, self._publish_mode)
            self.create_timer(STATUS_PERIOD_S, self._status_tick)

            self.get_logger().info(f"CSV log: {path}")
            self._log_event("START", "-", STOP, self.arb.stop_reason)
            self._publish_mode()

        # ---- callbacks ----
        def _on_manual(self, m):
            self.arb.on_manual(m.linear.x, m.angular.z, time.monotonic())

        def _on_auto(self, m):
            self.arb.on_auto(m.linear.x, m.angular.z, time.monotonic())

        def _on_view(self, m):
            self.arb.on_view(m.data, time.monotonic())

        def _on_request(self, m):
            self.arb.request(m.data, time.monotonic())

        # ---- timers ----
        def _control_tick(self):
            thr, steer = self.arb.step(time.monotonic())
            self._out = (thr, steer)
            cmd = Twist()
            cmd.linear.x = float(thr)
            cmd.angular.z = float(steer)
            self.pub_drive.publish(cmd)

        def _publish_mode(self):
            self.pub_mode.publish(String(data=self.arb.mode))

        def _status_tick(self):
            now = time.monotonic()
            a = self.arb
            age = a._age(a._active_sample(), now) if a.mode != STOP else float("nan")
            extra = f"reason='{a.stop_reason}'" if a.mode == STOP else \
                f"src={SOURCE_FOR_MODE[a.mode]} age={age:.2f}s view_ok={a.view_desc(now)}"
            self.get_logger().info(
                f"STATUS mode={a.mode} in_control={CONTROLLER_FOR_MODE[a.mode]} "
                f"out=(thr {self._out[0]:+.2f}, steer {self._out[1]:+.2f}) {extra}")
            self._write_row("STATUS", a.mode, a.mode, extra)

        # ---- logging ----
        def _write_row(self, event, frm, to, reason):
            self._csv.writerow([datetime.now().isoformat(timespec="milliseconds"),
                                f"{time.monotonic() - self._t0:.3f}", event, frm, to,
                                CONTROLLER_FOR_MODE[self.arb.mode], reason,
                                f"{self._out[0]:.3f}", f"{self._out[1]:.3f}"])
            self._csv_f.flush()

        def _log_event(self, event, frm, to, reason):
            msg = (f"{event}: {frm} -> {to} | in control: "
                   f"{CONTROLLER_FOR_MODE[self.arb.mode]} | why: {reason}")
            if event in ("STOP_LATCHED", "REJECTED"):
                self.get_logger().warn(msg)
            else:
                self.get_logger().info(msg)
            self._write_row(event, frm, to, reason)
            if event != "REJECTED":
                self._publish_mode()

        def shutdown(self):
            self._log_event("SHUTDOWN", self.arb.mode, STOP, "control_node exiting")
            for _ in range(5):
                self.pub_drive.publish(Twist())
                time.sleep(0.02)
            self._csv_f.close()

    rclpy.init()
    node = ControlNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
