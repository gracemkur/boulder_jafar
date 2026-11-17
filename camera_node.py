#!/usr/bin/env python3
"""
FloodBot perception node (Person A).

Contract (do not rename):
  /perception/line_offset  std_msgs/Float32  -1.0 far left .. 0.0 center .. +1.0 far right, NaN = no line
  /perception/view_ok      std_msgs/Bool     True = image usable, False = too blurred / dark / occluded
  /perception/status       std_msgs/String   JSON, one per frame:
      {"offset": float|null, "view_ok": bool, "lap_var": float, "bright": float,
       "source": "color"|"ir", "confidence": float, "degrade": int, "reason": str}
  /perception/degrade      std_msgs/Int32    (subscribed) set the artificial blur step live, 0..10
  (status/degrade/debug_image are extras, not part of the contract)
  /perception/debug_image/compressed  sensor_msgs/CompressedImage  annotated JPEG for the dashboard
      (tape outline, offset, view_ok, source). Only encoded while something subscribes. --image-hz 0 = off.

Modes:
  python3 camera_node.py                 # ROS 2 node, ~15 Hz (source /opt/ros/humble/setup.bash first)
  python3 camera_node.py --test          # no ROS: grab one frame, print status fields, save debug.jpg
  python3 camera_node.py --test --image some.jpg   # no ROS, no camera: run detection on a saved image
  python3 camera_node.py --test --image some.jpg --dark-sim 0.2   # pretend the room is 5x darker

How the line is found (same detector for color and IR):
  1. Downscale to 320x240, keep the bottom ROI (--roi-top), box-blur to wash out carpet texture
     (IR: median filter first to kill the emitter dot pattern).
  2. Floor model: per-row median brightness (handles the near/far brightness gradient), spread = MAD
     of the residual. Each pixel gets a contrast score z = (pixel - row median) / spread
     (sign flipped for --line dark). Pixels with z > --z-thresh are tape candidates. This is relative
     to the floor, so it works whether the room is bright or dim.
  3. Candidates must look like tape: elongated along the driving direction (principal axis within
     --max-angle of vertical), span a good part of the ROI height (--min-span), reach the bottom of
     the ROI or a side edge, plausible width (--max-width), reasonably solid (fill of minAreaRect),
     and a minimum absolute contrast (--min-contrast gray levels). Carpet streaks and sideways
     objects (e.g. a paper roll lying across the view) fail these tests.
  4. confidence 0..1 = contrast + span + shape score of the best blob. offset = NaN if confidence
     < --min-conf. offset = blob centroid x, normalized to the analysed image width.

Source selection (color vs infrared):
  The D435 IR stream 1 (left imager, 640x480 y8) runs alongside color. When the color image is dark
  (mean < --ir-enter-bright), or the color line confidence is low while the room is dim
  (mean < --ir-lowconf-bright), for --switch-frames frames, the node switches to IR. It goes back to
  color only after --return-frames frames with color mean > --ir-exit-bright and either a good color
  line or plenty of light. If IR itself becomes unusable while color is fine, it goes back to color.
  NOTE: the IR left imager and the color camera have different fields of view and a few cm of
  baseline. They are roughly aligned horizontally and offset is normalized per image, so -1..+1 means
  "left edge .. right edge of the image in use"; expect a small bias (a few hundredths) when switching.
  Re-check auto_node gains on IR if steering looks off in the dark.

view_ok (debounced):
  Raw check = the ACTIVE source passes its blur (Laplacian variance) and brightness window
  (color: --blur-min/--bright-min/--bright-max, IR: --ir-blur-min/--ir-bright-min/--ir-bright-max).
  view_ok goes False after --bad-frames consecutive bad frames, back to True after --good-frames
  consecutive good ones. Starts False. Missing frame / camera timeout => NaN + False immediately.

Tuning guide:
  * Brighter room / glare: if view_ok drops with reason "washed-out", raise --bright-max. If the tape
    is found but confidence is low, lower --z-thresh (e.g. 2.5) or --min-conf (e.g. 0.3).
  * Darker room: the detector is relative, so first just watch the "source" field. If it switches to
    IR too early/late, move --ir-enter-bright (switch to IR below this) and --ir-exit-bright (back to
    color above this; keep it >= enter + 15 to avoid flicker). If color view_ok goes False with
    "dark" before IR kicks in, lower --bright-min or raise --ir-enter-bright above --bright-min.
  * Carpet patches detected as line: raise --z-thresh (3.5-4), --min-span (0.4) or --min-contrast.
  * Tape missed in curves (line leaves through the side): raise --max-angle (70) / lower --min-span.
  * IR thresholds (UNTESTED on real IR frames - check with --test --save-raw in the dark):
    --ir-bright-min (default 8) / --ir-bright-max (245) / --ir-blur-min (30). The emitter dots raise
    the IR Laplacian variance, so --ir-blur-min is less sensitive than the color one. Try
    --ir-emitter off if the dot pattern breaks the tape blob (opt-in: on this Pi changing the emitter
    has timed out and once wedged the USB).
  * --thresh N forces the old fixed gray threshold on the COLOR image (-1 = Otsu); IR stays adaptive.
  * Pi 4 speed: everything after the blur/brightness check runs on a 320x240 image; the log line
    prints the processing ms. --debug-every -1 disables debug.jpg writes.

Useful options (work in every mode):
  --line light|dark      light tape on dark floor (default) or dark tape on light floor
  --degrade N            artificially blur the ACTIVE source by N steps (G2 poor-view test)
  --ramp SECS            (ROS mode) step degrade 0,1,..,N every SECS seconds, then repeat
                         (a message on /perception/degrade overrides --degrade/--ramp)
  --source auto|color|ir force a source (default auto)
  --ir-emitter on|off|leave   try once at startup to set the IR emitter (default leave = don't touch)
  --bad-frames N / --good-frames M   view_ok debounce (default 3 / 3)
  --dark-sim F [--dark-noise S]      (--test) scale brightness by F and add Gaussian noise sigma S
  --ir-image FILE        (--test --image) pair an IR frame with the color image
"""

import argparse
import array
import json
import math
import os
import sys
import time

import cv2
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DEBUG_PATH = os.path.join(SCRIPT_DIR, "debug.jpg")
WIDTH, HEIGHT, FPS = 640, 480, 15
SCALE = 0.5                              # analysis resolution = 320x240
AW, AH = int(WIDTH * SCALE), int(HEIGHT * SCALE)
DEGRADE_MAX = 10

# kernels allocated once (no per-frame allocations of structuring elements)
K_OPEN = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
K_CLOSE = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))


# --------------------------------------------------------------------------- camera

def import_realsense():
    """Import pyrealsense2, handling the source-build layout where the real module is nested."""
    try:
        import pyrealsense2 as rs
        if not hasattr(rs, "pipeline"):
            # Source builds sometimes install a package dir without the re-export in __init__.py
            from pyrealsense2 import pyrealsense2 as rs  # noqa: F811
        return rs
    except ImportError as e:
        sys.stderr.write(
            "\n[camera_node] Could not import pyrealsense2: %s\n"
            "  Fix: find the compiled module and put its folder on PYTHONPATH, e.g.\n"
            "    find / -name 'pyrealsense2*.so' 2>/dev/null\n"
            "    export PYTHONPATH=$PYTHONPATH:/usr/local/lib/python3.10/site-packages   # use the folder found above\n"
            "  and add that export line to ~/.bashrc. Make sure you run with python3 (not a venv or sudo).\n"
            "  Use --test --image file.jpg to test detection without the camera.\n\n" % e
        )
        raise SystemExit(1)


class RealSenseCamera:
    """Color (bgr8) + optional infrared stream 1 (y8), both 640x480 @ 15 fps."""

    def __init__(self, width=WIDTH, height=HEIGHT, fps=FPS, use_ir=True, ir_emitter="leave", log=None):
        rs = import_realsense()
        self.rs = rs
        self.log = log or (lambda msg: sys.stderr.write("[camera_node] %s\n" % msg))
        self.pipeline = rs.pipeline()
        self.ir_enabled = False
        profile = None
        if use_ir:
            cfg = rs.config()
            cfg.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
            cfg.enable_stream(rs.stream.infrared, 1, width, height, rs.format.y8, fps)
            try:
                profile = self.pipeline.start(cfg)
                self.ir_enabled = True
            except RuntimeError as e:
                self.log("could not start color+IR (%s); falling back to color only, no dark fallback" % e)
                self.pipeline = rs.pipeline()
        if profile is None:
            cfg = rs.config()
            cfg.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
            try:
                profile = self.pipeline.start(cfg)
            except RuntimeError as e:
                sys.stderr.write(
                    "[camera_node] Failed to start RealSense color stream: %s\n"
                    "  - Is another process using the camera (realsense-viewer, realsense2_camera node)?\n"
                    "  - Is the D435 on a USB3 (blue) port? Check: rs-enumerate-devices -s\n" % e
                )
                raise SystemExit(1)
        if self.ir_enabled and ir_emitter in ("on", "off"):
            self._set_emitter(profile, 1.0 if ir_emitter == "on" else 0.0)

    def _set_emitter(self, profile, value):
        """One attempt only. On this Pi set_xu has timed out and once wedged the USB, hence opt-in."""
        try:
            sensor = profile.get_device().first_depth_sensor()
            if sensor.supports(self.rs.option.emitter_enabled):
                sensor.set_option(self.rs.option.emitter_enabled, value)
                self.log("IR emitter set to %s" % ("on" if value else "off"))
            else:
                self.log("IR emitter option not supported; left as is")
        except Exception as e:  # noqa: BLE001 - any failure here must not kill the node
            self.log("could not set IR emitter (%s); leaving it as is" % e)

    def warmup(self, n=30):
        """Drop frames so auto-exposure settles (important for single-shot test mode)."""
        for _ in range(n):
            self.pipeline.wait_for_frames(5000)

    def read(self, timeout_ms=1000):
        """Return (color_bgr or None, ir_gray or None). Raises RuntimeError on timeout."""
        frames = self.pipeline.wait_for_frames(timeout_ms)
        color = frames.get_color_frame()
        c = np.asanyarray(color.get_data()).copy() if color else None
        ir = None
        if self.ir_enabled:
            f = frames.get_infrared_frame(1)
            if f:
                ir = np.asanyarray(f.get_data()).copy()
        return c, ir

    def stop(self):
        try:
            self.pipeline.stop()
        except Exception:
            pass


# --------------------------------------------------------------------------- vision

def degrade(img, steps):
    """Artificial blur for G2. Each step widens the Gaussian kernel by 4 px (step 1 = 5x5, 5 = 21x21)."""
    if steps <= 0 or img is None:
        return img
    k = 4 * steps + 1
    return cv2.GaussianBlur(img, (k, k), 0)


def dark_sim(img, factor, noise, rng):
    """Simulate a darker room: scale brightness and add Gaussian sensor noise."""
    if img is None or factor is None:
        return img
    f = img.astype(np.float32) * factor
    if noise > 0:
        f += rng.normal(0.0, noise, img.shape).astype(np.float32)
    return np.clip(f, 0, 255).astype(np.uint8)


def gray_mean(img):
    """Mean luminance of a gray or BGR image without a full color conversion."""
    m = cv2.mean(img)
    return float(m[0]) if img.ndim == 2 else 0.114 * m[0] + 0.587 * m[1] + 0.299 * m[2]


def view_stats(gray):
    """(laplacian_variance, mean_brightness) on the full-resolution gray image."""
    lap = cv2.Laplacian(gray, cv2.CV_16S)
    _, sd = cv2.meanStdDev(lap)
    return float(sd[0, 0]) ** 2, float(cv2.mean(gray)[0])


def check_view(lap_var, bright, src, p):
    """Return list of reasons the view is bad for this source (empty = ok)."""
    if src == "ir":
        bmin, lo, hi = p.ir_blur_min, p.ir_bright_min, p.ir_bright_max
    else:
        bmin, lo, hi = p.blur_min, p.bright_min, p.bright_max
    reasons = []
    if lap_var < bmin:
        reasons.append("blurry")
    if bright < lo:
        reasons.append("dark")
    if bright > hi:
        reasons.append("washed-out")
    return reasons


def _clip01(x):
    return 0.0 if x < 0.0 else 1.0 if x > 1.0 else float(x)


def find_line(gray_small, p, is_ir=False):
    """
    Adaptive tape detector on the 320x240 gray image.
    Returns dict(offset, confidence, contour (analysis coords, full image), centroid, top, thresh, why).
    """
    h, w = gray_small.shape
    top = int(h * p.roi_top)
    roi = gray_small[top:, :]
    if is_ir:
        roi = cv2.medianBlur(roi, 5)               # IR emitter dots -> gone
    sm = cv2.blur(roi, (7, 7))                     # carpet texture -> mostly gone
    smf = sm.astype(np.float32)
    rh = roi.shape[0]

    # floor model: per-row median (near rows are brighter/darker than far rows)
    rowmed = np.median(smf, axis=1)
    resid = smf - rowmed[:, None]
    if p.line == "dark":
        resid = -resid
    spread = max(float(np.median(np.abs(resid))) * 1.4826, p.min_spread)
    z = resid * (1.0 / spread)

    use_fixed = (not is_ir) and p.thresh is not None
    if use_fixed:
        mode = cv2.THRESH_BINARY_INV if p.line == "dark" else cv2.THRESH_BINARY
        if p.thresh < 0:
            t, mask = cv2.threshold(sm, 0, 255, mode | cv2.THRESH_OTSU)
        else:
            t, mask = cv2.threshold(sm, p.thresh, 255, mode)
        thresh_desc = "fixed %.0f" % t
    else:
        mask = (z > p.z_thresh).astype(np.uint8) * 255
        thresh_desc = "z>%.1f" % p.z_thresh

    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, K_OPEN)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, K_CLOSE)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    roi_area = float(rh * w)
    min_area = p.min_area * SCALE * SCALE
    best, best_conf, best_why = None, 0.0, "no-line"
    blob_mask = np.zeros_like(mask)

    def centerline_angle(c):
        """Angle from vertical of the line through each row's middle (robust for short, wide pieces)."""
        blob_mask[:] = 0
        cv2.drawContours(blob_mask, [c], -1, 255, -1)
        ys, xs = np.nonzero(blob_mask)
        rows = np.unique(ys)
        if rows.size < 3:
            return 90.0
        centers = np.array([xs[ys == r].mean() for r in rows])
        slope = np.polyfit(rows.astype(np.float64), centers, 1)[0]  # dx per dy
        return abs(math.degrees(math.atan(slope)))

    def shape(c, reached_before=False, trimmed=False):
        """Shape stats + pass/fail for one contour (analysis coords inside the ROI)."""
        a = cv2.contourArea(c)
        x, y, bw, bh = cv2.boundingRect(c)
        span = bh / float(rh)
        reaches = reached_before or (y + bh >= rh * (1.0 - p.bottom_gap)) or x <= 1 or x + bw >= w - 1
        m = cv2.moments(c)
        if m["m00"] <= 0:
            return None
        if trimmed:
            # every row of a trimmed piece is already tape-width, so its centre line is the tape direction
            ang = centerline_angle(c)
        else:
            # principal axis angle measured from vertical (0 = straight ahead, 90 = across the view)
            theta = 0.5 * math.atan2(2.0 * m["mu11"], m["mu20"] - m["mu02"])  # from x axis
            ang = abs(90.0 - abs(math.degrees(theta)))
        mean_w = a / max(bh, 1)
        (_, _), (rw, rhh), _ = cv2.minAreaRect(c)
        fill = a / max(rw * rhh, 1.0)
        ok = reaches and span >= p.min_span and ang <= p.max_angle and mean_w <= p.max_width * w \
            and fill >= p.min_fill
        return dict(ok=ok, a=a, span=span, ang=ang, fill=fill, m=m, reaches=reaches)

    def trim_wide_rows(c):
        """Tape touching another light object (a paper roll, a wall edge) merges into one blob that
        fails the shape test. Drop the rows where the blob is wider than tape can be and keep the
        largest remaining piece; the caller re-tests its shape."""
        blob_mask[:] = 0
        cv2.drawContours(blob_mask, [c], -1, 255, -1)
        widths = np.count_nonzero(blob_mask, axis=1)
        wide = widths > p.max_row_width * w
        if not wide.any():
            return None
        blob_mask[wide, :] = 0
        cs, _ = cv2.findContours(blob_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cs = [k for k in cs if cv2.contourArea(k) >= min_area]
        return max(cs, key=cv2.contourArea) if cs else None

    for c in contours:
        a = cv2.contourArea(c)
        if a < min_area:
            continue
        if a > p.max_area_frac * roi_area:
            best_why = "blob-too-big"
            continue
        s = shape(c)
        if s is None:
            continue
        trimmed = False
        if not s["ok"]:
            t = trim_wide_rows(c)
            s2 = shape(t, reached_before=s["reaches"], trimmed=True) if t is not None else None
            if s2 is None or not s2["ok"]:
                if best is None:
                    best_why = "no-line-shape"
                continue
            c, s, trimmed = t, s2, True
        a, span, ang, fill, m = s["a"], s["span"], s["ang"], s["fill"], s["m"]
        blob_mask[:] = 0
        cv2.drawContours(blob_mask, [c], -1, 255, -1)
        zc = cv2.mean(z, mask=blob_mask)[0]
        dc = cv2.mean(resid, mask=blob_mask)[0]      # gray levels above the floor
        if dc < p.min_contrast or zc < p.z_thresh:
            if best is None:
                best_why = "no-line-contrast"
            continue
        contrast_s = _clip01((zc - p.z_thresh) / max(p.z_good - p.z_thresh, 1e-3))
        span_s = _clip01(span / 0.7)
        angle_s = _clip01(1.0 - max(0.0, ang - 30.0) / max(p.max_angle - 30.0, 1.0))
        fill_s = _clip01(fill / 0.7)
        conf = 0.2 + 0.4 * contrast_s + 0.2 * span_s + 0.1 * angle_s + 0.1 * fill_s
        if trimmed:
            conf *= 0.85  # part of the blob was cut away: trust it a bit less
        conf = _clip01(conf)
        if conf > best_conf:
            best, best_conf = c, conf
            best_m = m

    res = dict(offset=float("nan"), confidence=0.0, contour=None, centroid=None,
               top=top, thresh=thresh_desc, why=best_why, spread=spread)
    if best is None:
        return res
    cx = best_m["m10"] / best_m["m00"]
    cy = best_m["m01"] / best_m["m00"] + top
    res["confidence"] = best_conf
    res["contour"] = best + np.array([[[0, top]]], dtype=best.dtype)
    res["centroid"] = (cx, cy)
    if best_conf < p.min_conf:
        res["why"] = "low-conf"
        return res
    off = (cx - w / 2.0) / (w / 2.0)
    if is_ir:
        off *= p.ir_offset_scale  # IR imager sees wider than color: rescale so offsets match at a switch
    res["offset"] = max(-1.0, min(1.0, off))
    res["why"] = "line"
    return res


def to_small_gray(img):
    g = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return g, cv2.resize(g, (AW, AH), interpolation=cv2.INTER_AREA)


# --------------------------------------------------------------------------- state

class Debounce:
    """view_ok debounce: False after n_bad bad frames, True after n_good good frames."""

    def __init__(self, n_bad, n_good, state=False):
        self.n_bad, self.n_good = max(1, n_bad), max(1, n_good)
        self.state, self.bad, self.good = state, 0, 0

    def update(self, ok):
        if ok:
            self.good, self.bad = self.good + 1, 0
            if not self.state and self.good >= self.n_good:
                self.state = True
        else:
            self.bad, self.good = self.bad + 1, 0
            if self.state and self.bad >= self.n_bad:
                self.state = False
        return self.state

    def force_bad(self):
        self.state, self.bad, self.good = False, self.n_bad, 0


class Perception:
    """Per-frame pipeline: source selection (with hysteresis), detection, view check, debounce."""

    def __init__(self, p, single_shot=False):
        self.p = p
        self.single = single_shot
        self.source = "ir" if p.source == "ir" else "color"
        self.count = 0                 # consecutive frames wanting the other source
        self.deb = Debounce(p.bad_frames, p.good_frames)
        self.switch_msg = None

    def _want_switch(self, color_bright, color_conf, ir_ok, ir_avail, color_view_ok):
        p = self.p
        if p.source != "auto":
            return False
        if self.source == "color":
            if not ir_avail or not ir_ok:
                return False
            if color_bright < p.ir_enter_bright:
                return True
            return color_conf is not None and color_conf < p.min_conf and color_bright < p.ir_lowconf_bright
        # on IR
        if not ir_avail:
            return True
        if not ir_ok and color_view_ok:
            return True
        if color_bright > p.ir_exit_bright and color_conf is not None:
            return color_conf >= p.min_conf or color_bright > p.ir_lowconf_bright + 10
        return False

    def step(self, color, ir, steps):
        p = self.p
        self.switch_msg = None
        ir_avail = ir is not None
        if p.source == "ir" and ir_avail:
            self.source = "ir"
        if not ir_avail and self.source == "ir":
            self.source = "color"
            self.switch_msg = "IR frame missing -> color"

        if self.single and p.source == "auto" and ir_avail:
            # single frame: decide directly (no history)
            cb = gray_mean(color)
            if cb < p.ir_enter_bright:
                self.source = "ir"

        def analyse(src):
            img = degrade(color if src == "color" else ir, steps)
            g, gs = to_small_gray(img)
            lv, br = view_stats(g)
            det = find_line(gs, p, is_ir=(src == "ir"))
            return img, lv, br, det

        frame, lap_var, bright, det = analyse(self.source)
        view_bad = check_view(lap_var, bright, self.source, p)

        if self.single and p.source == "auto" and ir_avail and self.source == "color" \
                and det["confidence"] < p.min_conf and bright < p.ir_lowconf_bright:
            self.source = "ir"
            frame, lap_var, bright, det = analyse("ir")
            view_bad = check_view(lap_var, bright, "ir", p)

        used = self.source             # source this frame's result came from (reported in status)
        # inputs for the hysteresis decision about the OTHER source
        if self.source == "color":
            color_bright, color_conf, color_view_ok = bright, det["confidence"], not view_bad
            ir_ok = ir_avail and p.ir_bright_min <= gray_mean(ir) <= p.ir_bright_max
        else:
            color_bright = gray_mean(color) if color is not None else 0.0
            ir_ok = not view_bad
            color_conf, color_view_ok = None, color_bright >= p.bright_min
            if color is not None and color_bright > p.ir_exit_bright:
                color_conf = find_line(to_small_gray(degrade(color, steps))[1], p)["confidence"]

        if not self.single:
            if self._want_switch(color_bright, color_conf, ir_ok, ir_avail, color_view_ok):
                self.count += 1
                need = p.switch_frames if self.source == "color" else p.return_frames
                if self.source == "ir" and (not ir_avail or not ir_ok):
                    need = p.switch_frames
                if self.count >= need:
                    old = self.source
                    self.source = "ir" if old == "color" else "color"
                    self.count = 0
                    self.switch_msg = "source %s -> %s (color bright %.0f, conf %s)" % (
                        old, self.source, color_bright,
                        "n/a" if color_conf is None else "%.2f" % color_conf)
            else:
                self.count = 0

        raw_ok = not view_bad
        view_ok = raw_ok if self.single else self.deb.update(raw_ok)
        reasons = list(view_bad)
        if det["why"] != "line":
            reasons.append(det["why"])
        if view_ok != raw_ok:
            reasons.append("settling")
        info = dict(offset=det["offset"], view_ok=bool(view_ok), raw_ok=raw_ok, lap_var=lap_var,
                    bright=bright, source=used, confidence=det["confidence"],
                    degrade=int(steps), reason=",".join(reasons) or "ok", thresh=det["thresh"])
        return frame, info, det

    def missing(self, steps, why="no-frame"):
        self.deb.force_bad()
        return dict(offset=float("nan"), view_ok=False, raw_ok=False, lap_var=0.0, bright=0.0,
                    source=self.source, confidence=0.0, degrade=int(steps), reason=why, thresh="-")


def status_json(info):
    off = info["offset"]
    return json.dumps({
        "offset": None if off is None or not math.isfinite(off) else round(float(off), 4),
        "view_ok": bool(info["view_ok"]),
        "lap_var": round(float(info["lap_var"]), 1),
        "bright": round(float(info["bright"]), 1),
        "source": info["source"],
        "confidence": round(float(info["confidence"]), 3),
        "degrade": int(info["degrade"]),
        "reason": str(info["reason"]),
    })


# --------------------------------------------------------------------------- debug output

def draw_debug(frame, info, det):
    img = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR) if frame.ndim == 2 else frame.copy()
    h, w = img.shape[:2]
    inv = 1.0 / SCALE
    top = int(det["top"] * inv)
    cv2.rectangle(img, (0, top), (w - 1, h - 1), (255, 200, 0), 1)          # ROI
    cv2.line(img, (w // 2, top), (w // 2, h), (200, 200, 200), 1)           # image center
    if det["contour"] is not None:
        good = math.isfinite(info["offset"])
        cnt = (det["contour"].astype(np.float32) * inv).astype(np.int32)
        cv2.drawContours(img, [cnt], -1, (0, 255, 0) if good else (0, 165, 255), 2)
        cx, cy = int(det["centroid"][0] * inv), int(det["centroid"][1] * inv)
        cv2.circle(img, (cx, cy), 7, (0, 0, 255), -1)
        cv2.line(img, (w // 2, cy), (cx, cy), (0, 0, 255), 2)
    off = info["offset"]
    off_s = "NaN (no line)" if not math.isfinite(off) else "%+.2f" % off
    ok_col = (0, 200, 0) if info["view_ok"] else (0, 0, 255)
    lines = [
        ("offset: %s  conf: %.2f" % (off_s, info["confidence"]), (255, 255, 255)),
        ("view_ok: %s  source: %s" % (info["view_ok"], info["source"].upper()), ok_col),
        ("reason: %s" % info["reason"], ok_col),
        ("lapvar: %.0f  bright: %.0f  %s  degrade: %d"
         % (info["lap_var"], info["bright"], info["thresh"], info["degrade"]), (255, 255, 255)),
    ]
    cv2.rectangle(img, (0, 0), (w - 1, 8 + 22 * len(lines)), (0, 0, 0), -1)  # text panel
    for i, (txt, col) in enumerate(lines):
        cv2.putText(img, txt, (8, 22 + i * 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 1, cv2.LINE_AA)
    return img


def save_debug(img, path):
    """Write atomically so nobody copies a half-written jpg."""
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    tmp = path + ".tmp.jpg"
    if cv2.imwrite(tmp, img):
        os.replace(tmp, path)


def summary(info):
    off = info["offset"]
    return "offset=%s view_ok=%s (%s) src=%s conf=%.2f lapvar=%.1f bright=%.1f %s degrade=%d" % (
        "nan" if not math.isfinite(off) else "%+.3f" % off, info["view_ok"], info["reason"],
        info["source"], info["confidence"], info["lap_var"], info["bright"], info["thresh"],
        info["degrade"])


# --------------------------------------------------------------------------- modes

def _load(path, gray=False):
    img = cv2.imread(os.path.expanduser(path), cv2.IMREAD_GRAYSCALE if gray else cv2.IMREAD_COLOR)
    if img is None:
        raise SystemExit("[camera_node] could not read image %s" % path)
    if img.shape[:2] != (HEIGHT, WIDTH):
        img = cv2.resize(img, (WIDTH, HEIGHT))
    return img


def run_test(p):
    ir = None
    if p.image:
        frame = _load(p.image)
        if p.ir_image:
            ir = _load(p.ir_image, gray=True)
    else:
        cam = RealSenseCamera(use_ir=p.source != "color", ir_emitter=p.ir_emitter)
        try:
            cam.warmup()
            frame, ir = cam.read(5000)
        finally:
            cam.stop()
        if frame is None:
            raise SystemExit("[camera_node] no color frame received")
        if p.save_raw:
            d = os.path.dirname(p.debug_path)
            save_debug(frame, os.path.join(d, "raw.jpg"))
            if ir is not None:
                save_debug(ir, os.path.join(d, "raw_ir.jpg"))

    if p.dark_sim is not None:
        rng = np.random.default_rng(0)
        frame = dark_sim(frame, p.dark_sim, p.dark_noise, rng)  # color only: IR is emitter-lit

    per = Perception(p, single_shot=True)
    if p.source == "ir" and ir is None:
        raise SystemExit("[camera_node] --source ir needs an IR frame (camera or --ir-image)")
    t0 = time.perf_counter()
    shown, info, det = per.step(frame, ir, max(0, min(DEGRADE_MAX, p.degrade)))
    ms = (time.perf_counter() - t0) * 1000.0
    save_debug(draw_debug(shown, info, det), p.debug_path)
    print(status_json(info))
    print(summary(info) + "  (%.1f ms)" % ms)
    print("debug image saved to %s" % p.debug_path)
    return info


IMAGE_TOPIC = "/perception/debug_image/compressed"


def publish_debug_image(node, pub, img, p):
    """JPEG-encode the annotated frame and publish it as sensor_msgs/CompressedImage (for the dashboard)."""
    from sensor_msgs.msg import CompressedImage
    h, w = img.shape[:2]
    if p.image_width < w:
        img = cv2.resize(img, (p.image_width, int(h * p.image_width / w)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, p.image_quality])
    if not ok:
        return
    msg = CompressedImage()
    msg.header.stamp = node.get_clock().now().to_msg()
    msg.header.frame_id = "camera"
    msg.format = "jpeg"
    msg.data = array.array("B", buf.tobytes())
    pub.publish(msg)


def run_ros(p):
    import rclpy
    from rclpy.node import Node
    from std_msgs.msg import Bool, Float32, Int32, String

    rclpy.init()
    node = Node("camera_node")
    pub_off = node.create_publisher(Float32, "/perception/line_offset", 10)
    pub_ok = node.create_publisher(Bool, "/perception/view_ok", 10)
    pub_status = node.create_publisher(String, "/perception/status", 10)
    pub_img = None
    if p.image_hz > 0:
        from rclpy.qos import QoSProfile, ReliabilityPolicy
        from sensor_msgs.msg import CompressedImage
        pub_img = node.create_publisher(CompressedImage, IMAGE_TOPIC,
                                        QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT))
    last_img = 0.0
    log = node.get_logger()

    live = {"degrade": None}  # set by /perception/degrade; overrides --degrade/--ramp

    def on_degrade(msg):
        v = max(0, min(DEGRADE_MAX, int(msg.data)))
        if v != live["degrade"]:
            log.info("degrade set live: %s -> %d%s" % (live["degrade"], v,
                     " (clamped from %d)" % msg.data if v != msg.data else ""))
        live["degrade"] = v

    node.create_subscription(Int32, "/perception/degrade", on_degrade, 10)

    cam = RealSenseCamera(use_ir=p.source != "color", ir_emitter=p.ir_emitter,
                          log=lambda m: log.warn(m))
    log.info("camera started %dx%d@%d ir=%s, line=%s thresh=%s z=%.1f min_conf=%.2f blur_min=%.0f "
             "bright=[%.0f,%.0f] ir_bright=[%.0f,%.0f] ir_enter/exit=%.0f/%.0f degrade=%d ramp=%s"
             % (WIDTH, HEIGHT, FPS, cam.ir_enabled, p.line, p.thresh, p.z_thresh, p.min_conf,
                p.blur_min, p.bright_min, p.bright_max, p.ir_bright_min, p.ir_bright_max,
                p.ir_enter_bright, p.ir_exit_bright, p.degrade, p.ramp))

    per = Perception(p)
    t0 = time.monotonic()
    last_debug = 0.0
    last_log = 0.0
    proc_ms = 0.0
    try:
        while rclpy.ok():
            steps = p.degrade
            if p.ramp > 0 and p.degrade > 0:
                steps = int((time.monotonic() - t0) // p.ramp) % (p.degrade + 1)
            if live["degrade"] is not None:
                steps = live["degrade"]
            steps = max(0, min(DEGRADE_MAX, steps))

            why = "no-color-frame"
            try:
                frame, ir = cam.read(1000)
            except RuntimeError as e:  # frame timeout: tell drivers we can't see
                log.warn("camera read failed: %s" % e)
                frame, ir, why = None, None, "timeout"
            now = time.monotonic()

            if frame is None:  # safe path: NaN + False immediately, no debounce
                info = per.missing(steps, why)
                pub_off.publish(Float32(data=float("nan")))
                pub_ok.publish(Bool(data=False))
                pub_status.publish(String(data=status_json(info)))
                rclpy.spin_once(node, timeout_sec=0.0)
                continue

            tp = time.perf_counter()
            shown, info, det = per.step(frame, ir, steps)
            proc_ms = 0.9 * proc_ms + 0.1 * (time.perf_counter() - tp) * 1000.0
            pub_off.publish(Float32(data=float(info["offset"])))
            pub_ok.publish(Bool(data=bool(info["view_ok"])))
            pub_status.publish(String(data=status_json(info)))
            if per.switch_msg:
                log.info(per.switch_msg)

            if (pub_img is not None and now - last_img >= 1.0 / p.image_hz
                    and node.count_subscribers(IMAGE_TOPIC) > 0):
                last_img = now
                publish_debug_image(node, pub_img, draw_debug(shown, info, det), p)

            if p.debug_every >= 0 and now - last_debug >= p.debug_every:
                save_debug(draw_debug(shown, info, det), p.debug_path)
                last_debug = now
            if now - last_log >= 1.0:
                log.info(summary(info) + " proc=%.1fms" % proc_ms)
                last_log = now

            rclpy.spin_once(node, timeout_sec=0.0)
    except KeyboardInterrupt:
        pass
    finally:
        cam.stop()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="FloodBot camera perception node")
    ap.add_argument("--test", action="store_true", help="no ROS: one frame, print result, save debug.jpg")
    ap.add_argument("--image", help="(with --test) use this image file instead of the camera")
    ap.add_argument("--ir-image", help="(with --test --image) IR image (gray) paired with --image")
    ap.add_argument("--save-raw", action="store_true",
                    help="(with --test) also save the raw frames as raw.jpg / raw_ir.jpg")
    ap.add_argument("--dark-sim", type=float, default=None,
                    help="(with --test) multiply COLOR brightness by this factor (e.g. 0.2) to simulate darkness; "
                         "an IR frame (camera or --ir-image) is left as is")
    ap.add_argument("--dark-noise", type=float, default=4.0,
                    help="(with --dark-sim) Gaussian noise sigma in gray levels (default 4)")
    ap.add_argument("--line", choices=["dark", "light"], default="light", help="tape color vs floor")
    ap.add_argument("--thresh", type=int, default=None,
                    help="override: fixed gray threshold 0-255 on the color image, -1 = Otsu "
                         "(default: adaptive contrast detector)")
    ap.add_argument("--z-thresh", type=float, default=3.0,
                    help="adaptive: tape pixels are this many floor-spreads brighter (darker) than the floor")
    ap.add_argument("--z-good", type=float, default=4.5, help="adaptive: mean contrast that counts as 'full confidence'")
    ap.add_argument("--min-spread", type=float, default=2.0, help="floor spread lower bound (gray levels)")
    ap.add_argument("--min-contrast", type=float, default=4.0,
                    help="min mean gray-level difference tape vs floor (rejects noise in flat images)")
    ap.add_argument("--min-conf", type=float, default=0.4, help="offset = NaN below this confidence")
    ap.add_argument("--roi-top", type=float, default=0.5,
                    help="only search below this fraction of image height (default 0.5 = bottom half)")
    ap.add_argument("--min-area", type=float, default=400, help="min tape blob area in full-res px")
    ap.add_argument("--max-area-frac", type=float, default=0.6,
                    help="reject blobs bigger than this fraction of the ROI (floor/shadow, not tape)")
    ap.add_argument("--min-span", type=float, default=0.3, help="blob must cover this fraction of ROI height")
    ap.add_argument("--bottom-gap", type=float, default=0.4,
                    help="blob must reach the bottom (this fraction) of the ROI or a side edge")
    ap.add_argument("--max-angle", type=float, default=60.0, help="max tilt of the tape axis from vertical (deg)")
    ap.add_argument("--max-width", type=float, default=0.45, help="max mean blob width as fraction of image width")
    ap.add_argument("--ir-offset-scale", type=float, default=1.6,
                    help="multiply IR offsets by this so they match color offsets (IR field of view is wider; "
                         "1.6 measured on the car: same tape gave +0.365 color vs +0.215 IR)")
    ap.add_argument("--max-row-width", type=float, default=0.25,
                    help="if a blob fails the shape test, drop rows wider than this (fraction of image width) "
                         "and re-test; recovers tape touching another light object")
    ap.add_argument("--min-fill", type=float, default=0.35, help="min blob area / minAreaRect area")
    ap.add_argument("--blur-min", type=float, default=60.0, help="color: Laplacian variance below this = blurry")
    ap.add_argument("--bright-min", type=float, default=30.0, help="color: mean gray below this = too dark")
    ap.add_argument("--bright-max", type=float, default=230.0, help="color: mean gray above this = washed out")
    ap.add_argument("--ir-blur-min", type=float, default=30.0, help="IR: Laplacian variance below this = blurry")
    ap.add_argument("--ir-bright-min", type=float, default=8.0, help="IR: mean below this = too dark")
    ap.add_argument("--ir-bright-max", type=float, default=245.0, help="IR: mean above this = washed out")
    ap.add_argument("--source", choices=["auto", "color", "ir"], default="auto",
                    help="image source: auto (IR fallback in the dark), color only, or IR only")
    ap.add_argument("--ir-enter-bright", type=float, default=45.0,
                    help="switch to IR when color mean brightness is below this")
    ap.add_argument("--ir-exit-bright", type=float, default=65.0,
                    help="go back to color only when color mean brightness is above this")
    ap.add_argument("--ir-lowconf-bright", type=float, default=70.0,
                    help="also switch to IR if color line confidence is low AND color mean is below this")
    ap.add_argument("--switch-frames", type=int, default=2, help="frames before switching color -> IR")
    ap.add_argument("--return-frames", type=int, default=15, help="frames before switching IR -> color")
    ap.add_argument("--ir-emitter", choices=["on", "off", "leave"], default="leave",
                    help="try once at startup to set the IR emitter (default leave: don't touch, safest)")
    ap.add_argument("--bad-frames", type=int, default=3, help="consecutive bad frames before view_ok=False")
    ap.add_argument("--good-frames", type=int, default=3, help="consecutive good frames before view_ok=True")
    ap.add_argument("--degrade", type=int, default=0, help="artificial blur steps (0 = off, max 10)")
    ap.add_argument("--ramp", type=float, default=0.0,
                    help="(ROS mode) cycle degrade 0..N, advancing one step every RAMP seconds")
    ap.add_argument("--debug-path", default=DEFAULT_DEBUG_PATH,
                    help="debug image path (default: debug.jpg next to this script)")
    ap.add_argument("--image-hz", type=float, default=10.0,
                    help="(ROS mode) max rate of %s for the dashboard; 0 = off" % IMAGE_TOPIC)
    ap.add_argument("--image-width", type=int, default=640, help="downscale the dashboard image to this width")
    ap.add_argument("--image-quality", type=int, default=70, help="JPEG quality 1-100 for the dashboard image")
    ap.add_argument("--debug-every", type=float, default=1.0,
                    help="(ROS mode) seconds between debug.jpg saves; -1 = never")
    # ignore ROS args (e.g. --ros-args ...) if someone passes them
    p, _ = ap.parse_known_args(argv)
    p.debug_path = os.path.expanduser(p.debug_path)
    return p


def main():
    p = parse_args()
    if p.test:
        run_test(p)
    else:
        run_ros(p)


if __name__ == "__main__":
    main()
