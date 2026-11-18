#!/usr/bin/env python3
"""
FloodBot autonomy node (Person B).

Contract (do not rename):
  subscribes /perception/line_offset  std_msgs/Float32  -1 far left .. 0 center .. +1 far right, NaN = no line
             /perception/view_ok      std_msgs/Bool     True = image usable
             /perception/status       std_msgs/String   (optional) camera_node JSON; only "confidence"
                                                        0..1 is used. Missing/stale/unparsable -> None.
             /mode                    std_msgs/String   control_node mode manual|auto|assist|stop
                                                        (reliable + transient-local, same QoS as control_node)
  publishes  /cmd/auto                geometry_msgs/Twist
                linear.x  = throttle [-1..1]  (+ forward, - reverse)
                angular.z = steer    [-1..1], +z = turn LEFT
             /cmd/mode_request        std_msgs/String   "stop:<why>" ONCE when the line is lost for good,
                                                        so control_node latches STOP and logs why (reliable)

The decision logic lives in AutoController (pure Python, no ROS imports at module level), so a
simulator / test can do:
    from auto_node import AutoController, parse_args
    ctl = AutoController(parse_args(["--throttle", "0.5"]))
    out = ctl.step(now, offset, view_ok, offset_age, ok_age, confidence=None, mode="auto")
    # out = dict(throttle, steer, reason, state, request)   request = None or "stop:..."
main() is only a thin ROS wrapper: it feeds the latest messages into step() at --rate Hz,
publishes the Twist, and publishes `request` (if any) on /cmd/mode_request.

Steering (recomputed only when a NEW camera sample arrives, held between samples):
  raw   = -k * offset - kd * d(offset)/dt          (dt = time between camera samples, derivative
                                                    skipped on the first sample after a lost line)
  steer = s * steer_prev + (1 - s) * raw           (s = --steer-smooth, 0 = no smoothing)
  clamped to +-max_steer.  Line right of center (offset > 0) -> negative z -> turn RIGHT, toward it.

Throttle (G2: the car visibly SLOWS as the view degrades and in curves):
  f_turn = 1 - turn_slow * |steer| / max_steer      (e.g. 0.4 -> 40 % slower at full steer)
  f_conf = 1 if confidence is None or >= conf_full,
           else min_throttle_frac + (1 - min_throttle_frac) * confidence / conf_full
  throttle = max(min_throttle, throttle * f_turn * f_conf)
  --min-throttle is a floor that still moves the car: drive_node adds --deadband-offset to every
  non-zero command, but with --deadband-offset 80 this car only rolls at about 0.5 (0.15 does not).
  Measure the real threshold with esc_sweep.py (wheels up) and set --min-throttle just above it.

Lost-line recovery state machine (state names are logged and returned by step()):
  tracking     line visible: drive as above. Recovery attempts reset to 0 only after tracking
               continuously for --reset-after s (a short regain does not refund an attempt).
  grace        offset NaN while view_ok True and perception fresh: first COAST for --lost-coast s
               (keep the last tracking throttle and steer: carries the car over short tape gaps and
               fork rejoins), then zero throttle holding the last steer for --lost-grace s.
               Line back -> tracking (a short blip costs nothing).
  brake        (only with --esc-reverse-dance on) --brake-time s of -brake_throttle, then
               --neutral-time s of zero: hobby ESCs need brake -> neutral -> reverse to reverse.
  reverse      throttle = -reverse_throttle, steer 0, for up to --reverse-time s, or until the line is
               seen again -> tracking.
  search-wait  --search-wait s of zero, then (if attempts remain) reverse again, else gave-up.
  gave-up      after --max-recoveries attempts (counted when brake/reverse starts): publish zero
               forever and return request "stop:auto lost line after N recoveries" exactly once, so
               control_node latches STOP and logs it. Never drives again on its own; only a human
               (R, then O: /mode re-enters "auto") starts a fresh session.
  This bounds the finish-line loop (tape ends -> back up -> see tape -> forward -> lose it ...) and
  fixes the old bug where the car drove off by itself when tape reappeared after a lost line: it may
  resume only inside the bounded recovery window, never after gave-up.
  Also (not recovery states):
  blind        view_ok False or perception stale (> --stale s): zero immediately, never reverse,
               abort any recovery (control_node latches STOP on view_ok anyway). Attempts are kept.
  idle         /mode is not "auto": zero, recovery state and attempts reset. Entering "auto" from
               any other mode (a human pressed O) also resets them. Until the first /mode message
               arrives the node assumes "auto" (bench tests without control_node).
  --max-recoveries 0 disables reversing: lost line -> grace -> gave-up.

Publishes a ZERO Twist on shutdown.

Flags (defaults):
  --k 1.0  --kd 0.05  --steer-smooth 0.3  --max-steer 1.0
  --throttle 0.15  --min-throttle 0.0  --turn-slow 0.4  --conf-full 0.7  --min-throttle-frac 0.5
  --stale 0.5  --rate 20
  --lost-coast 0.5  --lost-grace 0.3  --esc-reverse-dance on  --brake-throttle 0.3  --brake-time 0.15  --neutral-time 0.15
  --reverse-throttle 0.5  --reverse-time 1.0  --search-wait 0.3  --max-recoveries 2  --reset-after 3.0

Usage:
  source /opt/ros/humble/setup.bash
  python3 auto_node.py [flags]
  Team run line on Jafara (drive_node --deadband-offset 80; --throttle 0.15 default is kept only for
  backwards compatibility and does NOT move this car):
    AUTO_ARGS="--throttle 0.5 --min-throttle 0.45" ./run.sh
  (0.45 is a placeholder: set --min-throttle to the lowest throttle that still rolls the car.)
"""

import argparse
import json
import math
import signal
import sys
import time

TRACKING, GRACE, BRAKE, REVERSE, SEARCH_WAIT, GAVE_UP = (
    "tracking", "grace", "brake", "reverse", "search-wait", "gave-up")
BLIND, IDLE = "blind", "idle"


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="FloodBot line follower with lost-line recovery -> /cmd/auto")
    ap.add_argument("--k", type=float, default=1.0, help="steering gain: steer = -k * offset")
    ap.add_argument("--kd", type=float, default=0.05,
                    help="derivative gain on d(offset)/dt between camera samples, seconds (default 0.05)")
    ap.add_argument("--steer-smooth", type=float, default=0.3,
                    help="steer low-pass per camera sample, 0 = off .. <1 (default 0.3)")
    ap.add_argument("--max-steer", type=float, default=1.0, help="clamp |steer| to this (default 1.0)")
    ap.add_argument("--throttle", type=float, default=0.15,
                    help="base forward throttle [0..1] (default 0.15; Jafara needs ~0.5)")
    ap.add_argument("--min-throttle", type=float, default=0.0,
                    help="never scale forward throttle below this (keep above the ESC deadband)")
    ap.add_argument("--turn-slow", type=float, default=0.4,
                    help="fraction of throttle removed at full steer, 0..1 (default 0.4)")
    ap.add_argument("--conf-full", type=float, default=0.7,
                    help="perception confidence at/above which there is no slow-down (default 0.7)")
    ap.add_argument("--min-throttle-frac", type=float, default=0.5,
                    help="throttle fraction at confidence 0 (default 0.5)")
    ap.add_argument("--stale", type=float, default=0.5, help="seconds before perception counts as dead")
    ap.add_argument("--rate", type=float, default=20.0, help="publish rate in Hz")
    ap.add_argument("--lost-coast", type=float, default=0.5,
                    help="seconds to keep the last throttle + steering when the line vanishes while the view is "
                         "good (crosses short tape gaps and fork rejoins; 0 = stop at once). Bounded: then "
                         "grace/recovery as usual. Never used when view_ok is False or perception is stale")
    ap.add_argument("--lost-grace", type=float, default=0.3,
                    help="seconds of NaN (zero throttle, hold steer) before recovery starts")
    ap.add_argument("--esc-reverse-dance", choices=["on", "off"], default="on",
                    help="brake pulse -> neutral before reversing (default on)")
    ap.add_argument("--brake-throttle", type=float, default=0.3, help="brake pulse magnitude (default 0.3)")
    ap.add_argument("--brake-time", type=float, default=0.15, help="brake pulse seconds (default 0.15)")
    ap.add_argument("--neutral-time", type=float, default=0.15, help="neutral after brake, seconds (default 0.15)")
    ap.add_argument("--reverse-throttle", type=float, default=0.5,
                    help="reverse magnitude, goes through the same deadband mapping (default 0.5)")
    ap.add_argument("--reverse-time", type=float, default=1.0, help="max seconds reversing per attempt")
    ap.add_argument("--search-wait", type=float, default=0.3, help="seconds of zero after a failed reverse")
    ap.add_argument("--max-recoveries", type=int, default=2,
                    help="recovery attempts per AUTO session before giving up (default 2, 0 = never reverse)")
    ap.add_argument("--reset-after", type=float, default=3.0,
                    help="seconds of continuous tracking that reset the attempt counter (default 3.0)")
    p, _ = ap.parse_known_args(argv)  # ignore --ros-args etc.
    if not 0.0 <= p.throttle <= 1.0:
        ap.error("--throttle must be 0..1")
    if not 0.0 <= p.min_throttle <= p.throttle:
        ap.error("--min-throttle must be 0..--throttle")
    if not 0.0 < p.max_steer <= 1.0:
        ap.error("--max-steer must be (0, 1]")
    if p.stale <= 0 or p.rate <= 0:
        ap.error("--stale and --rate must be > 0")
    if p.kd < 0 or not 0.0 <= p.steer_smooth < 1.0:
        ap.error("--kd must be >= 0 and --steer-smooth 0..<1")
    if not 0.0 <= p.turn_slow <= 1.0 or not 0.0 <= p.min_throttle_frac <= 1.0 or not 0.0 < p.conf_full <= 1.0:
        ap.error("--turn-slow and --min-throttle-frac must be 0..1, --conf-full (0, 1]")
    if not (0.0 <= p.brake_throttle <= 1.0 and 0.0 <= p.reverse_throttle <= 1.0):
        ap.error("--brake-throttle and --reverse-throttle must be 0..1")
    if min(p.lost_coast, p.lost_grace, p.brake_time, p.neutral_time, p.reverse_time, p.search_wait, p.reset_after) < 0:
        ap.error("times must be >= 0")
    if p.max_recoveries < 0:
        ap.error("--max-recoveries must be >= 0")
    return p


def _finite(x):
    return x is not None and math.isfinite(x)


def throttle_for(steer, confidence, p):
    """Forward throttle for a given steer and confidence (None = unknown -> no slow-down)."""
    if p.throttle <= 0.0:
        return 0.0
    f = 1.0 - p.turn_slow * min(1.0, abs(steer) / p.max_steer)
    if _finite(confidence) and confidence < p.conf_full:
        c = max(0.0, confidence) / p.conf_full
        f *= p.min_throttle_frac + (1.0 - p.min_throttle_frac) * c
    return max(p.min_throttle, p.throttle * f)


def compute(offset, view_ok, offset_age, ok_age, p, confidence=None):
    """Stateless single-sample law (no kd, no smoothing, no recovery): (throttle, steer, reason).
    Zero command unless everything is fresh and valid."""
    if offset_age > p.stale or ok_age > p.stale:
        return 0.0, 0.0, "perception stale"
    if not view_ok:
        return 0.0, 0.0, "view_ok=False"
    if not _finite(offset):
        return 0.0, 0.0, "no line (NaN)"
    steer = max(-p.max_steer, min(p.max_steer, -p.k * offset))
    return throttle_for(steer, confidence, p), steer, "tracking"


class AutoController:
    """Line follower + bounded lost-line recovery. Pure Python, all times supplied by the caller.

    step(now, offset, view_ok, offset_age, ok_age, confidence=None, mode="auto")
      -> dict(throttle, steer, reason, state, request)
    Attributes worth logging: state, recoveries (attempts used this AUTO session).
    """

    def __init__(self, p):
        self.p = p
        self.mode = None
        self.reset()

    def reset(self):
        """Forget everything: back to tracking with zero attempts (new AUTO session)."""
        self.state = TRACKING
        self.recoveries = 0
        self.state_t = None        # when the current state was entered
        self.track_since = None    # start of the current continuous tracking stretch
        self.steer = 0.0           # last steer output (held in grace)
        self.track_thr = 0.0       # last tracking throttle (used while coasting over a gap)
        self.prev_sample = None    # (t, offset) of the last valid camera sample, for kd
        self.last_sample_t = None  # timestamp of the last camera sample seen (valid or NaN)
        self.request_sent = False

    # ---- helpers -----------------------------------------------------------
    def _enter(self, state, now):
        if state != self.state:
            self.state = state
            self.state_t = now
            if state != TRACKING:
                self.track_since = None

    def _out(self, thr, steer, reason, request=None):
        return dict(throttle=float(thr), steer=float(steer), reason=reason, state=self.state, request=request)

    def _update_steer(self, now, offset, offset_age):
        """Recompute steer only when a new camera sample arrived."""
        t = now - offset_age
        if self.last_sample_t is not None and abs(t - self.last_sample_t) < 1e-6:
            return
        self.last_sample_t = t
        p = self.p
        raw = -p.k * offset
        if self.prev_sample is not None and p.kd > 0:
            dt = t - self.prev_sample[0]
            if dt > 1e-3:
                raw -= p.kd * (offset - self.prev_sample[1]) / dt
        self.prev_sample = (t, offset)
        raw = max(-p.max_steer, min(p.max_steer, raw))
        s = p.steer_smooth
        self.steer = max(-p.max_steer, min(p.max_steer, s * self.steer + (1.0 - s) * raw))

    def _start_attempt(self, now, dance):
        """Start a recovery attempt if any remain, else give up (with a one-shot stop request)."""
        if self.recoveries >= self.p.max_recoveries:
            self._enter(GAVE_UP, now)
            self.steer = 0.0
            req = None
            if not self.request_sent:
                self.request_sent = True
                req = "stop:auto lost line after %d recoveries" % self.recoveries
            return self._out(0.0, 0.0, "gave up: line lost after %d recoveries" % self.recoveries, req)
        self.recoveries += 1
        self.steer = 0.0
        self._enter(BRAKE if dance else REVERSE, now)
        return None

    # ---- one control cycle -------------------------------------------------
    def step(self, now, offset, view_ok, offset_age, ok_age, confidence=None, mode="auto"):
        p = self.p
        prev_mode, self.mode = self.mode, mode
        if mode != "auto":
            if self.state != IDLE or self.recoveries:
                self.reset()
                self.state = IDLE
                self.state_t = now
            return self._out(0.0, 0.0, "mode=%s" % mode)
        if prev_mode is not None and prev_mode != "auto":
            self.reset()              # a human pressed O: fresh AUTO session
            self.state_t = now
        if self.state == IDLE:
            self._enter(TRACKING, now)
        if self.state_t is None:
            self.state_t = now

        if self.state == GAVE_UP:
            return self._out(0.0, 0.0, "gave up: line lost after %d recoveries (waiting for human)"
                             % self.recoveries)

        if offset_age > p.stale or ok_age > p.stale or not view_ok:
            self._enter(BLIND, now)
            self.prev_sample = None
            self.steer = 0.0
            return self._out(0.0, 0.0, "perception stale" if (offset_age > p.stale or ok_age > p.stale)
                             else "view_ok=False")

        if _finite(offset):
            if self.state != TRACKING:
                self._enter(TRACKING, now)
            if self.track_since is None:
                self.track_since = now
            if self.recoveries and now - self.track_since >= p.reset_after:
                self.recoveries = 0
            self._update_steer(now, offset, offset_age)
            self.track_thr = throttle_for(self.steer, confidence, p)
            return self._out(self.track_thr, self.steer, "tracking")

        # ---- line lost (NaN) with a good, fresh view ----
        self.prev_sample = None
        self.last_sample_t = now - offset_age
        if self.state in (TRACKING, BLIND):
            self._enter(GRACE, now)
        el = now - self.state_t

        if self.state == GRACE:
            if el < p.lost_coast:
                # keep rolling with the last throttle and steering: carries the car over short gaps
                return self._out(getattr(self, "track_thr", 0.0), self.steer, "no line: coasting %.2fs" % el)
            if el < p.lost_coast + p.lost_grace:
                return self._out(0.0, self.steer, "no line: grace %.2fs" % el)
            out = self._start_attempt(now, p.esc_reverse_dance == "on")
            if out:
                return out
            el = 0.0

        if self.state == BRAKE:
            if el < p.brake_time:
                return self._out(-p.brake_throttle, 0.0, "no line: brake pulse (attempt %d)" % self.recoveries)
            if el < p.brake_time + p.neutral_time:
                return self._out(0.0, 0.0, "no line: neutral before reverse (attempt %d)" % self.recoveries)
            self._enter(REVERSE, now)
            el = 0.0

        if self.state == REVERSE:
            if el < p.reverse_time:
                return self._out(-p.reverse_throttle, 0.0, "no line: reversing (attempt %d/%d)"
                                 % (self.recoveries, p.max_recoveries))
            self._enter(SEARCH_WAIT, now)
            el = 0.0

        if self.state == SEARCH_WAIT:
            if el < p.search_wait:
                return self._out(0.0, 0.0, "no line: search wait after attempt %d" % self.recoveries)
            # already stopped after reversing: an ESC goes straight back into reverse, no dance
            out = self._start_attempt(now, False)
            if out:
                return out
            return self._out(-p.reverse_throttle, 0.0, "no line: reversing (attempt %d/%d)"
                             % (self.recoveries, p.max_recoveries))

        return self._out(0.0, 0.0, "no line")  # not reached


def main():
    p = parse_args()

    import rclpy
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
    from geometry_msgs.msg import Twist
    from std_msgs.msg import Bool, Float32, String

    # Match control_node: /mode is reliable + transient-local (latched), /cmd/mode_request is reliable.
    qos_mode = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                          durability=DurabilityPolicy.TRANSIENT_LOCAL)
    qos_reliable = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE)

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
            self.ctl = AutoController(p)
            self.offset = None
            self.offset_t = None
            self.view_ok = False
            self.ok_t = None
            self.conf = None
            self.conf_t = None
            self.mode = None          # None until control_node's /mode arrives -> assume auto
            self.last = dict(throttle=0.0, steer=0.0, reason="starting", state=TRACKING, request=None)
            self.prev_state = None
            self.prev_recoveries = 0
            self.pub = self.create_publisher(Twist, "/cmd/auto", 10)
            self.pub_req = self.create_publisher(String, "/cmd/mode_request", qos_reliable)
            self.create_subscription(Float32, "/perception/line_offset", self.on_offset, 10)
            self.create_subscription(Bool, "/perception/view_ok", self.on_ok, 10)
            self.create_subscription(String, "/perception/status", self.on_status, 10)
            self.create_subscription(String, "/mode", self.on_mode, qos_mode)
            self.create_timer(1.0 / p.rate, self.tick)
            self.create_timer(1.0, self.log_tick)
            self.log.info("auto_node: k=%.3f kd=%.3f smooth=%.2f throttle=%.2f min_throttle=%.2f "
                          "turn_slow=%.2f conf_full=%.2f max_steer=%.2f stale=%.2fs rate=%.0fHz"
                          % (p.k, p.kd, p.steer_smooth, p.throttle, p.min_throttle, p.turn_slow,
                             p.conf_full, p.max_steer, p.stale, p.rate))
            self.log.info("recovery: grace=%.2fs dance=%s brake=%.2f/%.2fs neutral=%.2fs reverse=%.2f/%.2fs "
                          "search_wait=%.2fs max_recoveries=%d reset_after=%.1fs"
                          % (p.lost_grace, p.esc_reverse_dance, p.brake_throttle, p.brake_time,
                             p.neutral_time, p.reverse_throttle, p.reverse_time, p.search_wait,
                             p.max_recoveries, p.reset_after))

        def on_offset(self, msg):
            self.offset = float(msg.data)
            self.offset_t = time.monotonic()

        def on_ok(self, msg):
            self.view_ok = bool(msg.data)
            self.ok_t = time.monotonic()

        def on_status(self, msg):
            try:
                c = json.loads(msg.data).get("confidence")
                self.conf = float(c) if c is not None else None
            except (ValueError, TypeError, AttributeError):
                self.conf = None
            self.conf_t = time.monotonic()

        def on_mode(self, msg):
            if msg.data != self.mode:
                self.log.info("mode -> %s" % msg.data)
            self.mode = msg.data

        def publish(self, throttle, steer):
            t = Twist()
            t.linear.x = float(throttle)
            t.angular.z = float(steer)
            self.pub.publish(t)

        def tick(self):
            now = time.monotonic()
            off_age = float("inf") if self.offset_t is None else now - self.offset_t
            ok_age = float("inf") if self.ok_t is None else now - self.ok_t
            conf = self.conf if self.conf_t is not None and now - self.conf_t <= p.stale else None
            out = self.ctl.step(now, self.offset, self.view_ok, off_age, ok_age,
                                confidence=conf, mode=self.mode or "auto")
            self.publish(out["throttle"], out["steer"])
            self.last = out
            if out["request"]:
                self.pub_req.publish(String(data=out["request"]))
                self.log.warn("published /cmd/mode_request '%s'" % out["request"])
            if out["state"] != self.prev_state:
                (self.log.info if out["state"] == TRACKING else self.log.warn)(
                    "state -> %s (%s) recoveries=%d/%d"
                    % (out["state"], out["reason"], self.ctl.recoveries, p.max_recoveries))
                self.prev_state = out["state"]
            if self.ctl.recoveries != self.prev_recoveries:
                if self.ctl.recoveries == 0:
                    self.log.info("recovery counter reset (tracking %.1fs or new AUTO session)" % p.reset_after)
                self.prev_recoveries = self.ctl.recoveries

        def log_tick(self):
            o = self.last
            off = "none" if self.offset is None else ("nan" if math.isnan(self.offset) else "%+.3f" % self.offset)
            conf = "none" if self.conf is None else "%.2f" % self.conf
            self.log.info("[%s] %s mode=%s offset=%s conf=%s view_ok=%s recoveries=%d/%d -> throttle=%+.2f steer=%+.3f"
                          % (o["state"], o["reason"], self.mode or "(none, assuming auto)", off, conf,
                             self.view_ok, self.ctl.recoveries, p.max_recoveries, o["throttle"], o["steer"]))

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
