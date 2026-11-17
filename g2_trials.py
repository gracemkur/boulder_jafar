#!/usr/bin/env python3
"""
FloodBot G2 "handle a poor view" trial recorder  (evidence tool, publishes NO drive commands)

What it does
  For each blur step (baseline 0, then --steps, default 1,3,5) and each trial (--trials, default 3):
  sets the artificial blur on camera_node via /perception/degrade, waits for the operator, records
  perception / mode / drive topics for --duration seconds (or until Enter), asks the operator
  whether the car really followed the line, then classifies the trial and writes CSV + summary.md.

Usage (on the Pi, three terminals):
  1) bash run.sh start                      # drive (ARM=1 for real runs), cam, auto, control_node
  2) source /opt/ros/humble/setup.bash && python3 teleop_key.py      # R = resume, O = auto, SPACE = E-stop
  3) source /opt/ros/humble/setup.bash && python3 g2_trials.py [--steps 1,3,5] [--trials 3]
                                                [--duration 20] [--out DIR] [--no-baseline]
                                                [--no-degrade-control]
  Per trial: place the car at the start, press Enter here, switch to auto in teleop (R then O),
  let it run; press Enter here to end early (or wait --duration), then answer y/n + optional note.
  With --no-degrade-control the tool only records; set blur yourself, e.g. restart camera_node with
  CAM_ARGS="--degrade 3" (the tool tells you which level each step expects).
  Ctrl-C at any point saves everything recorded so far and writes the summary.
  Outputs (default ~/robo/g2_results/<timestamp>/): raw_s<step>_d<level>_t<trial>.csv per trial,
  trials.csv (one row per trial), summary.md (per-step table + totals, paste into the readout).

How errors are counted (code_flagged = /perception/view_ok went False OR /mode entered "stop"):
  MISSED PROBLEM   = operator says it could NOT follow the line (n) and the code never flagged,
                     or flagged only after the failure time the operator gave (+0.5 s grace).
  UNNECESSARY STOP = operator says it followed fine (y) but the code flagged / stopped anyway.

Safety: this tool NEVER publishes /cmd/* or mode requests. It only publishes /perception/degrade
(std_msgs/Int32, at 2 Hz) and resets it to 0 on exit.
"""

import argparse
import csv
import json
import math
import os
import queue
import sys
import threading
import time
from datetime import datetime

FLAG_GRACE_S = 0.5          # code flag may come this long after the operator's failure time
CAMERA_WAIT_S = 10.0        # warn if no camera message within this many seconds
DEGRADE_HZ = 2.0            # /perception/degrade re-publish rate
THR_EPS = 1e-3              # |throttle| below this counts as zero
THR_DROP = 0.05             # throttle reduced by at least this below its running max = "reduced"

# topic keys used inside samples
T_OFF, T_OK, T_STATUS, T_MODE, T_DRIVE, T_AUTO = (
    "/perception/line_offset", "/perception/view_ok", "/perception/status",
    "/mode", "/cmd/drive", "/cmd/auto")

RAW_COLUMNS = ["t", "topic", "offset", "view_ok", "mode", "throttle", "steer",
               "lap_var", "bright", "confidence", "source", "degrade", "reason"]

TRIAL_COLUMNS = [
    "step", "degrade", "trial", "start_time", "duration_s", "ended_by",
    "follow_ok", "fail_time_s", "note",
    "n_view_ok", "n_offset", "n_status", "n_mode", "n_drive", "n_auto",
    "view_bad_frac", "t_first_view_bad", "nan_offset_frac",
    "mode_at_start", "stop_entered", "t_stop",
    "mean_lap_var", "mean_bright", "mean_confidence", "ir_frac", "degrade_reported",
    "drive_max_thr", "drive_zero_frac", "drive_thr_reduced", "drive_thr_cut", "t_drive_thr_cut",
    "auto_zero_frac", "t_auto_zero",
    "code_flagged", "t_flag", "outcome", "raw_file",
]

CORRECT, MISSED, UNNECESSARY, UNKNOWN = "CORRECT", "MISSED_PROBLEM", "UNNECESSARY_STOP", "UNKNOWN"


# ============================ pure logic (no ROS) ============================

def _f(x):
    """float or None (None/NaN-safe)."""
    if x is None:
        return None
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def _mean(vals):
    vals = [v for v in vals if v is not None and math.isfinite(v)]
    return sum(vals) / len(vals) if vals else None


def _is_nan_offset(v):
    return v is None or not math.isfinite(v)


def parse_status(text):
    """Parse /perception/status JSON into a dict with all known keys (missing -> None)."""
    try:
        d = json.loads(text)
        if not isinstance(d, dict):
            d = {}
    except (ValueError, TypeError):
        d = {}
    vo = d.get("view_ok")
    deg = d.get("degrade")
    try:
        deg = int(deg) if deg is not None else None
    except (TypeError, ValueError):
        deg = None
    return {
        "offset": _f(d.get("offset")),
        "view_ok": None if vo is None else bool(vo),
        "lap_var": _f(d.get("lap_var")),
        "bright": _f(d.get("bright")),
        "confidence": _f(d.get("confidence")),
        "source": d.get("source"),
        "degrade": deg,
        "reason": d.get("reason"),
    }


def classify(follow_ok, code_flagged, t_flag, fail_time):
    """follow_ok: True/False/None (operator ground truth). Returns outcome string."""
    if follow_ok is None:
        return UNKNOWN
    if follow_ok:
        return UNNECESSARY if code_flagged else CORRECT
    if not code_flagged:
        return MISSED
    if fail_time is not None and t_flag is not None and t_flag > fail_time + FLAG_GRACE_S:
        return MISSED  # flagged, but only after the car had already failed
    return CORRECT


def compute_metrics(samples, duration, follow_ok=None, fail_time=None):
    """samples: list of (t_rel, topic, dict). Returns a dict of per-trial metrics.

    A sample at t=0 with topic /mode may carry the mode that was latched before the trial.
    """
    m = {"duration_s": duration}
    by = {k: [] for k in (T_OFF, T_OK, T_STATUS, T_MODE, T_DRIVE, T_AUTO)}
    for t, topic, d in samples:
        if topic in by:
            by[topic].append((t, d))
    for k, name in ((T_OK, "n_view_ok"), (T_OFF, "n_offset"), (T_STATUS, "n_status"),
                    (T_MODE, "n_mode"), (T_DRIVE, "n_drive"), (T_AUTO, "n_auto")):
        m[name] = len(by[k])
    status = [d for _, d in by[T_STATUS]]

    # view_ok: prefer the Bool topic, fall back to status JSON
    vo = [(t, d["view_ok"]) for t, d in by[T_OK] if d.get("view_ok") is not None]
    if not vo:
        vo = [(t, d["view_ok"]) for t, d in by[T_STATUS] if d.get("view_ok") is not None]
    if vo:
        m["view_bad_frac"] = sum(1 for _, ok in vo if not ok) / len(vo)
        bad = [t for t, ok in vo if not ok]
        m["t_first_view_bad"] = bad[0] if bad else None
    else:
        m["view_bad_frac"] = None
        m["t_first_view_bad"] = None

    offs = [d.get("offset") for _, d in by[T_OFF]]
    if not offs:
        offs = [d.get("offset") for d in status]
    m["nan_offset_frac"] = (sum(1 for v in offs if _is_nan_offset(v)) / len(offs)) if offs else None

    # mode: count only transitions INTO stop (a stop latched before the trial is not a flag)
    modes = [(t, d.get("mode")) for t, d in by[T_MODE]]
    m["mode_at_start"] = modes[0][1] if modes else None
    m["stop_entered"] = False
    m["t_stop"] = None
    prev = None
    for t, md in modes:
        if md == "stop" and prev is not None and prev != "stop":
            m["stop_entered"] = True
            m["t_stop"] = t
            break
        prev = md

    # status extras
    m["mean_lap_var"] = _mean([d.get("lap_var") for d in status])
    m["mean_bright"] = _mean([d.get("bright") for d in status])
    m["mean_confidence"] = _mean([d.get("confidence") for d in status])
    srcs = [d.get("source") for d in status if d.get("source") in ("color", "ir")]
    m["ir_frac"] = (sum(1 for s in srcs if s == "ir") / len(srcs)) if srcs else None
    degs = [d.get("degrade") for d in status if d.get("degrade") is not None]
    m["degrade_reported"] = max(set(degs), key=degs.count) if degs else None

    # throttle on /cmd/drive (what the car actually got) and /cmd/auto (what autonomy asked for)
    def thr_stats(entries):
        thr = [(t, d.get("throttle")) for t, d in entries if d.get("throttle") is not None]
        if not thr:
            return None, None, False, False, None
        vals = [v for _, v in thr]
        zero_frac = sum(1 for v in vals if abs(v) < THR_EPS) / len(vals)
        run_max, reduced, cut, t_cut = 0.0, False, False, None
        for t, v in thr:
            if run_max > THR_EPS and v <= run_max - THR_DROP:
                reduced = True
            if run_max > THR_EPS and abs(v) < THR_EPS and not cut:
                cut, t_cut = True, t
            run_max = max(run_max, v)
        return max(vals), zero_frac, reduced, cut, t_cut

    mx, zf, red, cut, tcut = thr_stats(by[T_DRIVE])
    m["drive_max_thr"], m["drive_zero_frac"] = mx, zf
    m["drive_thr_reduced"], m["drive_thr_cut"], m["t_drive_thr_cut"] = red, cut, tcut
    _, azf, _, acut, atcut = thr_stats(by[T_AUTO])
    m["auto_zero_frac"], m["t_auto_zero"] = azf, (atcut if acut else None)

    flags = [t for t in (m["t_first_view_bad"], m["t_stop"]) if t is not None]
    m["code_flagged"] = bool(flags)
    m["t_flag"] = min(flags) if flags else None
    m["follow_ok"] = follow_ok
    m["fail_time_s"] = fail_time
    m["outcome"] = classify(follow_ok, m["code_flagged"], m["t_flag"], fail_time)
    return m


def summarize(trials):
    """trials: list of metric dicts with 'step' and 'degrade'. Returns (rows, totals)."""
    groups = {}
    for tr in trials:
        groups.setdefault((tr["step"], tr["degrade"]), []).append(tr)

    def agg(trs):
        return {
            "trials": len(trs),
            "follow_ok": sum(1 for t in trs if t.get("follow_ok") is True),
            "answered": sum(1 for t in trs if t.get("follow_ok") is not None),
            "code_stops": sum(1 for t in trs if t.get("code_flagged")),
            "missed": sum(1 for t in trs if t.get("outcome") == MISSED),
            "unnecessary": sum(1 for t in trs if t.get("outcome") == UNNECESSARY),
            "mean_view_bad": _mean([t.get("view_bad_frac") for t in trs]),
            "mean_nan": _mean([t.get("nan_offset_frac") for t in trs]),
            "mean_lap_var": _mean([t.get("mean_lap_var") for t in trs]),
            "mean_t_flag": _mean([t.get("t_flag") for t in trs]),
        }

    rows = []
    for (step, deg) in sorted(groups):
        r = agg(groups[(step, deg)])
        r["step"], r["degrade"] = step, deg
        rows.append(r)
    totals = agg(trials)
    totals["step"], totals["degrade"] = "total", "-"
    return rows, totals


def _fmt(v, nd=2, pct=False):
    if v is None:
        return "-"
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, float):
        if not math.isfinite(v):
            return "nan"
        return ("%.0f%%" % (100 * v)) if pct else ("%.*f" % (nd, v))
    return str(v)


TABLE_HEADER = ["Step", "Degrade", "Trials", "Follow OK", "Code stops", "Missed problems",
                "Unnecessary stops", "view_ok False", "NaN offset", "Mean lap_var",
                "Mean time-to-flag (s)"]


def table_rows(rows, totals):
    out = []
    for r in rows + [totals]:
        bold = r is totals
        cells = [r["step"], r["degrade"], r["trials"],
                 "%d/%d" % (r["follow_ok"], r["answered"]), r["code_stops"], r["missed"],
                 r["unnecessary"], _fmt(r["mean_view_bad"], pct=True), _fmt(r["mean_nan"], pct=True),
                 _fmt(r["mean_lap_var"], 1), _fmt(r["mean_t_flag"], 1)]
        cells = [str(c) for c in cells]
        if bold:
            cells = ["**%s**" % c for c in cells]
        out.append(cells)
    return out


def markdown_table(rows, totals):
    lines = ["| " + " | ".join(TABLE_HEADER) + " |",
             "|" + "|".join("---" for _ in TABLE_HEADER) + "|"]
    for cells in table_rows(rows, totals):
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def text_table(rows, totals):
    body = [[c.replace("**", "") for c in cells] for cells in table_rows(rows, totals)]
    allr = [TABLE_HEADER] + body
    w = [max(len(r[i]) for r in allr) for i in range(len(TABLE_HEADER))]
    fmt = lambda r: "  ".join(c.rjust(w[i]) for i, c in enumerate(r))
    lines = [fmt(TABLE_HEADER), "  ".join("-" * x for x in w)]
    lines += [fmt(r) for r in body[:-1]] + ["  ".join("-" * x for x in w), fmt(body[-1])]
    return "\n".join(lines)


def summary_markdown(trials, meta):
    rows, totals = summarize(trials)
    md = ["# G2 poor-view trials", "",
          "- Run: %s" % meta.get("started", "?"),
          "- Degrade control: %s" % ("operator (manual --degrade)" if meta.get("no_degrade_control")
                                     else "/perception/degrade from g2_trials.py"),
          "- Trial length: %s s max (operator may end early)" % meta.get("duration", "?"),
          "- /perception/status received: %s" % ("yes" if any(t.get("n_status") for t in trials) else "no "
                                                 "(metrics from view_ok/offset only)"),
          "", markdown_table(rows, totals), "",
          "code flagged = `/perception/view_ok` went False or `/mode` entered `stop` during the trial. "
          "**Missed problem** = operator said the car could not follow the line and the code did not flag "
          "(or flagged > %.1f s after the reported failure time). **Unnecessary stop** = operator said it "
          "followed fine but the code flagged/stopped." % FLAG_GRACE_S,
          "", "## Trials", "",
          "| Step | Degrade | Trial | Dur (s) | Follow OK | Code flagged | t_flag (s) | view_ok False "
          "| NaN offset | Stop entered | Drive thr cut | Outcome | Note |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for t in trials:
        md.append("| %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (
            t["step"], t["degrade"], t["trial"], _fmt(t.get("duration_s"), 1),
            _fmt(t.get("follow_ok")), _fmt(t.get("code_flagged")), _fmt(t.get("t_flag"), 1),
            _fmt(t.get("view_bad_frac"), pct=True), _fmt(t.get("nan_offset_frac"), pct=True),
            _fmt(t.get("stop_entered")), _fmt(t.get("drive_thr_cut")), t.get("outcome", "-"),
            (t.get("note") or "").replace("|", "/")))
    return "\n".join(md) + "\n", text_table(rows, totals)


def sample_to_row(t, topic, d):
    row = {"t": "%.3f" % t, "topic": topic}
    for k in RAW_COLUMNS[2:]:
        v = d.get(k)
        if isinstance(v, float):
            v = "nan" if not math.isfinite(v) else "%.4f" % v
        row[k] = "" if v is None else v
    return row


def trial_to_row(m):
    row = {}
    for k in TRIAL_COLUMNS:
        v = m.get(k)
        if isinstance(v, bool):
            v = int(v)
        elif isinstance(v, float):
            v = "%.4f" % v if math.isfinite(v) else "nan"
        elif v is None:
            v = ""
        row[k] = v
    if m.get("follow_ok") is None:
        row["follow_ok"] = ""
    return row


def parse_steps(text):
    try:
        steps = [int(s) for s in text.split(",") if s.strip() != ""]
    except ValueError:
        raise argparse.ArgumentTypeError("--steps must be comma-separated integers, e.g. 1,3,5")
    if not steps or any(s < 0 for s in steps):
        raise argparse.ArgumentTypeError("--steps needs at least one level >= 0")
    return steps


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="FloodBot G2 poor-view trial recorder (no drive commands)")
    ap.add_argument("--steps", type=parse_steps, default=[1, 3, 5],
                    help="degrade levels for the steps, comma-separated (default 1,3,5)")
    ap.add_argument("--no-baseline", action="store_true", help="skip the baseline step (degrade 0)")
    ap.add_argument("--trials", type=int, default=3, help="trials per step (default 3)")
    ap.add_argument("--duration", type=float, default=20.0,
                    help="max seconds per trial (default 20; Enter ends a trial early)")
    ap.add_argument("--out", default=None,
                    help="output directory (default ~/robo/g2_results/<timestamp>/)")
    ap.add_argument("--no-degrade-control", action="store_true",
                    help="don't publish /perception/degrade; operator sets blur (camera_node --degrade N)")
    p, _ = ap.parse_known_args(argv)  # ignore --ros-args etc.
    if p.trials < 1 or p.duration <= 0:
        ap.error("--trials must be >= 1 and --duration > 0")
    if p.out is None:
        p.out = os.path.join(os.path.expanduser("~/robo/g2_results"),
                             datetime.now().strftime("%Y%m%d_%H%M%S"))
    p.out = os.path.expanduser(p.out)
    levels = list(p.steps)
    if not p.no_baseline and 0 not in levels[:1]:
        levels = [0] + levels
    p.levels = levels  # step index i -> degrade level
    return p


# ============================ terminal input ================================

class LineReader:
    """One daemon thread owns stdin, so a timed-out trial never swallows the next answer."""

    def __init__(self):
        self.q = queue.Queue()
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        while True:
            line = sys.stdin.readline()
            if line == "":
                self.q.put(None)  # EOF
                return
            self.q.put(line.rstrip("\r\n"))

    def drain(self):
        """Discard type-ahead (e.g. stray Enters) but keep an EOF marker."""
        while True:
            try:
                line = self.q.get_nowait()
            except queue.Empty:
                return
            if line is None:
                self.q.put(None)
                return

    def get(self, timeout=None):
        """Wait for a line. Returns None on timeout; raises EOFError on EOF. Ctrl-C-interruptible."""
        end = None if timeout is None else time.monotonic() + timeout
        while True:
            wait = 0.2 if end is None else max(0.0, min(0.2, end - time.monotonic()))
            try:
                line = self.q.get(timeout=wait)
            except queue.Empty:
                if end is not None and time.monotonic() >= end:
                    return None
                continue
            if line is None:
                raise EOFError
            return line

    def ask(self, prompt):
        sys.stdout.write(prompt)
        sys.stdout.flush()
        return self.get()


# ================================ ROS side ==================================

def make_node(no_degrade_control):
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
    from geometry_msgs.msg import Twist
    from std_msgs.msg import Bool, Float32, Int32, String

    # BEST_EFFORT subscriptions match both reliable and best-effort publishers.
    qos_in = QoSProfile(depth=50, reliability=ReliabilityPolicy.BEST_EFFORT,
                        history=HistoryPolicy.KEEP_LAST)
    # /mode is RELIABLE + TRANSIENT_LOCAL in control_node.py: match it to get the latched value.
    qos_mode = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                          durability=DurabilityPolicy.TRANSIENT_LOCAL)

    class G2Recorder(Node):
        def __init__(self):
            super().__init__("g2_trials")
            self.lock = threading.Lock()
            self.recording = False
            self.t0 = None
            self.samples = []
            self.last_cam = None        # monotonic time of last camera message (any perception topic)
            self.last_status = None
            self.mode = None
            self.latest = {}            # latest dict per topic, for the live line
            self.degrade_level = 0
            self.pub_degrade = None
            if not no_degrade_control:
                self.pub_degrade = self.create_publisher(Int32, "/perception/degrade", 10)
                self.create_timer(1.0 / DEGRADE_HZ, self.publish_degrade)

            self.create_subscription(Float32, T_OFF, self._on_off, qos_in)
            self.create_subscription(Bool, T_OK, self._on_ok, qos_in)
            self.create_subscription(String, T_STATUS, self._on_status, qos_in)
            self.create_subscription(String, T_MODE, self._on_mode, qos_mode)
            self.create_subscription(Twist, T_DRIVE, self._on_drive, qos_in)
            self.create_subscription(Twist, T_AUTO, self._on_auto, qos_in)

        # ---- degrade (the ONLY thing this node publishes) ----
        def publish_degrade(self):
            if self.pub_degrade is not None:
                self.pub_degrade.publish(Int32(data=int(self.degrade_level)))

        def set_degrade(self, level):
            self.degrade_level = int(level)
            self.publish_degrade()

        # ---- recording ----
        def _add(self, topic, d, cam=False):
            now = time.monotonic()
            with self.lock:
                if cam:
                    self.last_cam = now
                self.latest[topic] = d
                if self.recording:
                    self.samples.append((now - self.t0, topic, d))

        def start(self):
            with self.lock:
                self.samples = []
                self.t0 = time.monotonic()
                self.recording = True
                if self.mode is not None:   # latched mode at trial start
                    self.samples.append((0.0, T_MODE, {"mode": self.mode}))

        def stop(self):
            with self.lock:
                self.recording = False
                return list(self.samples), time.monotonic() - self.t0

        # ---- callbacks ----
        def _on_off(self, m):
            self._add(T_OFF, {"offset": float(m.data)}, cam=True)

        def _on_ok(self, m):
            self._add(T_OK, {"view_ok": bool(m.data)}, cam=True)

        def _on_status(self, m):
            self.last_status = time.monotonic()
            self._add(T_STATUS, parse_status(m.data), cam=True)

        def _on_mode(self, m):
            self.mode = m.data
            self._add(T_MODE, {"mode": m.data})

        def _on_drive(self, m):
            self._add(T_DRIVE, {"throttle": float(m.linear.x), "steer": float(m.angular.z)})

        def _on_auto(self, m):
            self._add(T_AUTO, {"throttle": float(m.linear.x), "steer": float(m.angular.z)})

        def live_line(self):
            with self.lock:
                lt = dict(self.latest)
            ok = lt.get(T_OK, {}).get("view_ok")
            off = lt.get(T_OFF, {}).get("offset")
            st = lt.get(T_STATUS, {})
            thr = lt.get(T_DRIVE, {}).get("throttle")
            return "view_ok=%s offset=%s mode=%s drive_thr=%s conf=%s lap_var=%s" % (
                _fmt(ok), _fmt(off, 3), self.mode or "?", _fmt(thr), _fmt(st.get("confidence")),
                _fmt(st.get("lap_var"), 0))

    return G2Recorder()


# ================================== main ====================================

def main():
    p = parse_args()
    os.makedirs(p.out, exist_ok=True)

    import rclpy
    from rclpy.executors import MultiThreadedExecutor

    # Keep Python's KeyboardInterrupt (so Ctrl-C reaches input() in the main thread) and keep the
    # ROS context alive long enough to reset /perception/degrade to 0 on the way out.
    try:
        from rclpy.signals import SignalHandlerOptions
        rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
    except (ImportError, TypeError):
        rclpy.init()

    node = make_node(p.no_degrade_control)
    executor = MultiThreadedExecutor(num_threads=2)
    executor.add_node(node)
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()

    reader = LineReader()
    meta = {"started": datetime.now().isoformat(timespec="seconds"),
            "no_degrade_control": p.no_degrade_control, "duration": p.duration}
    trials = []

    def save_all():
        path = os.path.join(p.out, "trials.csv")
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=TRIAL_COLUMNS)
            w.writeheader()
            for t in trials:
                w.writerow(trial_to_row(t))
        md, txt = summary_markdown(trials, meta)
        with open(os.path.join(p.out, "summary.md"), "w") as f:
            f.write(md)
        return txt

    def finish_trial(step, level, k, samples, dur, ended_by, follow_ok, fail_time, note):
        raw_name = "raw_s%d_d%d_t%d.csv" % (step, level, k)
        with open(os.path.join(p.out, raw_name), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=RAW_COLUMNS)
            w.writeheader()
            for t, topic, d in samples:
                w.writerow(sample_to_row(t, topic, d))
        m = compute_metrics(samples, dur, follow_ok, fail_time)
        m.update({"step": step, "degrade": level, "trial": k, "ended_by": ended_by, "note": note,
                  "raw_file": raw_name,
                  "start_time": datetime.fromtimestamp(time.time() - dur).isoformat(timespec="seconds")})
        trials.append(m)
        save_all()
        print("  -> %s | flagged=%s t_flag=%s view_ok False %s, NaN %s, stop=%s, drive thr cut=%s" % (
            m["outcome"], _fmt(m["code_flagged"]), _fmt(m["t_flag"], 1),
            _fmt(m["view_bad_frac"], pct=True), _fmt(m["nan_offset_frac"], pct=True),
            _fmt(m["stop_entered"]), _fmt(m["drive_thr_cut"])))
        if m["n_view_ok"] == 0 and m["n_status"] == 0:
            print("  WARNING: no camera messages recorded in this trial (is camera_node running?)")
        if (not p.no_degrade_control and m["degrade_reported"] is not None
                and m["degrade_reported"] != level):
            print("  WARNING: camera_node reports degrade=%s but this step requested %d"
                  % (m["degrade_reported"], level))
        return m

    print("G2 trials -> %s" % p.out)
    print("steps (degrade levels): %s, %d trial(s) each, %.0f s max per trial"
          % (", ".join(map(str, p.levels)), p.trials, p.duration))
    print("This tool publishes ONLY /perception/degrade. Drive the modes with teleop_key.py.")

    # discovery on the Pi can be slow: wait for camera traffic before starting
    if not p.no_degrade_control:
        node.set_degrade(p.levels[0])
    t_wait = time.monotonic()
    while node.last_cam is None and time.monotonic() - t_wait < CAMERA_WAIT_S:
        time.sleep(0.1)
    if node.last_cam is None:
        print("WARNING: no /perception/* messages within %.0f s. Is camera_node running? "
              "(Recording anyway; discovery may still be in progress.)" % CAMERA_WAIT_S)
    elif node.last_status is None:
        print("note: no /perception/status yet; metrics will use view_ok/offset only if it never arrives.")

    current = None   # (step, level, k) while recording
    pending = None   # recorded trial waiting for the operator's answers
    aborted = False
    try:
        for step, level in enumerate(p.levels):
            print("\n=== Step %d: degrade %d ===" % (step, level))
            if p.no_degrade_control:
                print("Set the blur yourself now (e.g. restart camera_node with --degrade %d)." % level)
            else:
                node.set_degrade(level)
            for k in range(1, p.trials + 1):
                reader.drain()
                reader.ask("[step %d / degrade %d / trial %d/%d] Place car at start, press Enter to begin trial "
                           % (step, level, k, p.trials))
                if node.last_cam is None or time.monotonic() - node.last_cam > 2.0:
                    print("  WARNING: no camera message in the last 2 s.")
                if node.mode == "stop":
                    print("  note: /mode is STOP; press R then O in teleop to start auto.")
                node.start()
                current = (step, level, k)
                print("  RECORDING for up to %.0f s (Enter = end now, Ctrl-C = save and quit)" % p.duration)
                t_start = time.monotonic()
                ended_by = "timeout"
                while True:
                    left = p.duration - (time.monotonic() - t_start)
                    if left <= 0:
                        break
                    line = reader.get(timeout=min(1.0, left))
                    if line is not None:
                        ended_by = "operator"
                        break
                    sys.stdout.write("\r  t=%4.1fs %s   " % (time.monotonic() - t_start, node.live_line()))
                    sys.stdout.flush()
                samples, dur = node.stop()
                current = None
                pending = (step, level, k, samples, dur, ended_by)
                print("\n  trial ended (%s) after %.1f s, %d samples" % (ended_by, dur, len(samples)))

                ans = None
                while ans not in ("y", "n"):
                    ans = (reader.ask("  Did the car stay in the lane / could it follow the line? (y/n) ")
                           or "").strip().lower()[:1]
                fail_time = None
                if ans == "n":
                    ft = (reader.ask("  Roughly when did it fail, seconds into the trial? (Enter = unknown) ")
                          or "").strip()
                    try:
                        fail_time = float(ft) if ft else None
                    except ValueError:
                        fail_time = None
                note = (reader.ask("  Note (optional): ") or "").strip()
                pending = None
                finish_trial(step, level, k, samples, dur, ended_by, ans == "y", fail_time, note)
    except (KeyboardInterrupt, EOFError):
        aborted = True
        print("\nInterrupted: saving what was recorded so far ...")
        try:
            if current is not None:
                samples, dur = node.stop()
                finish_trial(current[0], current[1], current[2], samples, dur, "aborted", None, None,
                             "aborted (Ctrl-C), no ground truth")
            elif pending is not None:
                finish_trial(*pending, None, None, "recorded, but no ground truth (Ctrl-C)")
        except Exception as e:  # never lose the summary because of one trial
            print("  could not save the aborted trial: %s" % e)
    finally:
        try:
            txt = save_all()
            print("\n" + txt)
            print("\nwrote %s (trials.csv, summary.md, raw_*.csv)%s"
                  % (p.out, " [partial run]" if aborted else ""))
        except Exception as e:
            print("ERROR writing summary: %s" % e)
        if not p.no_degrade_control:
            node.degrade_level = 0
            for _ in range(5):
                try:
                    node.publish_degrade()
                except Exception:
                    break
                time.sleep(0.1)
            print("/perception/degrade reset to 0")
        executor.shutdown(timeout_sec=1.0)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
