#!/usr/bin/env python3
"""
Offline simulator for Jafara (FloodBot line follower) - tests the REAL team code without the car.

What it does
  * Renders what the RealSense D435 colour camera would see (640x480, ~69 deg HFOV, ~0.12 m high,
    pitched down so the horizon sits near y=100) looking at light beige paper tape (5 cm) on dark
    grey-green carpet, from a top-down ground texture (4 px/cm) and a kinematic bicycle car model.
  * Runs camera_node.Perception (the real detector + view_ok debounce + draw_debug) at 15 Hz with
    one frame of latency, auto_node.AutoController (the real controller + lost-line recovery) at
    20 Hz, and control_node.Arbiter (the real safety arbiter: STOP latch on view_ok False/stale,
    watchdog, "stop:..." requests) every sim step (60 Hz). Startup mimics a human: R (resume ->
    manual) and then O (auto) as soon as the arbiter accepts it (view_ok True and /cmd/auto fresh).
  * Falls back to a small adapter around auto_node.compute() if AutoController does not exist, and to
    a minimal arbiter if control_node.Arbiter cannot be imported.
  * Writes a side-by-side MP4 per run (annotated camera view | top-down map + mode/state/throttle/
    steer), results.csv, a per-run trace CSV, and prints a summary table.

How to run (needs numpy + opencv, no ROS):
  python3 sim.py --course all                      # every course, video for each
  python3 sim.py --course all --no-video           # fast
  python3 sim.py --course gap --throttle 0.6 --auto-args "--k 1.2 --max-recoveries 3"
  python3 sim.py --course fork2 --fork-plan left,right
  python3 sim.py --course straight --dark 0.3 --degrade 2 --cam-args "--z-thresh 2.5"
  Outputs go to ./sim_out/<timestamp>/ next to this file (or --out DIR).

Courses (--course): straight, oval (one counter-clockwise lap), s-curve, gap (25 cm gap in the tape:
  exercises lost-line recovery), end (tape just ends: correct = stop, not loop forever), fork2 (stem ->
  fork 1 whose two branches rejoin -> stem -> fork 2 whose branches end in separate end zones; the
  fork plan, default "left,right", says which branch to take at each fork), all.

Metrics (results.csv / summary table)
  outcome    finished    car reached the end zone of the course (oval: one full lap)
             stopped     control_node STOP latched (safety rule or the controller's "stop:" request)
                         and the car came to rest
             gave-up     still in AUTO but the car has not moved for 3 s (controller holds zero)
             left-course lateral error > 25 cm
             timeout     --duration reached;  no-start  the arbiter never accepted AUTO
  pass       course-specific success: finished for straight/oval/s-curve/gap; for end: stopped or
             gave-up with the car centre no more than 0.75 m past the tape end; for fork2: finished
             in the end zone the plan asks for with every fork choice as planned.
  time_s     sim seconds from AUTO engaged to the end of the run
  dist_m     distance along the (expected) course centreline from the start position
  odo_m      odometer (|v| integrated)
  max/mean_lat_cm  lateral distance of the car centre (rear axle + wheelbase/2) from the nearest tape
             centreline (gaps and the tape end are bridged by an invisible "ghost" centreline)
  reverses   number of controller state transitions into a reverse state (AutoController "reverse")
  forks      fork choices taken vs expected (fork2 only)
  stop_reason  control_node stop reason or the controller's last reason

Known limitations
  * Colour camera only: no IR stream is rendered, so camera_node's IR fallback is never exercised
    (--dark only scales the colour image; Perception gets ir=None).
  * Idealised kinematics: kinematic bicycle (no slip), first-order speed lag, linear deadband throttle
    map, a simple ESC brake->neutral->reverse model (--esc-reverse-dance), first-order steering servo.
  * Approximate camera calibration (pinhole, guessed height/pitch/FOV, no lens distortion, no motion
    blur, no rolling shutter); ground texture/lighting are synthetic (no glare, shadows, chair legs).
  * Timing is deterministic (no ROS jitter, no dropped frames); processing time is not simulated.
"""

import argparse
import csv
import inspect
import math
import os
import shlex
import sys
import time
from datetime import datetime

import cv2
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

import camera_node  # noqa: E402  (ROS imports inside functions only)
import auto_node    # noqa: E402

try:
    import control_node  # noqa: E402
    HAVE_ARBITER = hasattr(control_node, "Arbiter")
except Exception:  # noqa: BLE001
    control_node = None
    HAVE_ARBITER = False

W_IMG, H_IMG = 640, 480
SIM_HZ = 60
CAM_EVERY = 4          # 60 / 4 = 15 Hz camera
PPM = 400.0            # texture pixels per metre (4 px/cm)
DETAIL_PPM = 1000.0     # fine carpet detail tile, pixels per metre
PPM_REF = 200.0        # lateral-error raster pixels per metre
STEP = 0.005           # course polyline sampling (m)
COURSES = ["straight", "oval", "s-curve", "gap", "end", "fork2"]

CARPET_BGR = (94.0, 124.0, 117.0)
TAPE_BGR = (165.0, 185.0, 195.0)


# =============================================================================== geometry

def seg_path(start, heading, segs, step=STEP):
    """Polyline from segments: ("S", length) straight, ("A", radius, angle_rad) arc (+ = left)."""
    x, y = start
    h = heading
    pts = [(x, y)]
    for s in segs:
        if s[0] == "S":
            n = max(1, int(math.ceil(s[1] / step)))
            ds, k = s[1] / n, 0.0
        else:
            length = abs(s[2]) * s[1]
            n = max(1, int(math.ceil(length / step)))
            ds, k = length / n, math.copysign(1.0 / s[1], s[2])
        for _ in range(n):
            hm = h + 0.5 * ds * k
            x += ds * math.cos(hm)
            y += ds * math.sin(hm)
            h += ds * k
            pts.append((x, y))
    return np.array(pts, dtype=np.float64)


def bezier(p0, p1, p2, p3, step=STEP):
    p0, p1, p2, p3 = (np.asarray(p, dtype=np.float64) for p in (p0, p1, p2, p3))
    approx = np.linalg.norm(p1 - p0) + np.linalg.norm(p2 - p1) + np.linalg.norm(p3 - p2)
    n = max(2, int(approx / step * 2))
    t = np.linspace(0.0, 1.0, n)[:, None]
    pts = ((1 - t) ** 3) * p0 + 3 * ((1 - t) ** 2) * t * p1 + 3 * (1 - t) * t * t * p2 + t ** 3 * p3
    return resample(pts)


def resample(pts, step=STEP):
    d = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))]
    if d[-1] <= 0:
        return pts
    s = np.linspace(0.0, d[-1], max(2, int(d[-1] / step) + 1))
    return np.c_[np.interp(s, d, pts[:, 0]), np.interp(s, d, pts[:, 1])]


def join(*parts):
    out = [parts[0]]
    for p in parts[1:]:
        out.append(p[1:] if np.allclose(p[0], out[-1][-1], atol=1e-6) else p)
    return resample(np.vstack(out))


def cumlen(pts):
    return np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))]


def heading_at(pts, i):
    j = min(len(pts) - 1, i + 2)
    i0 = max(0, j - 4)
    d = pts[j] - pts[i0]
    return math.atan2(d[1], d[0])


FORK_ANGLE = math.radians(25.0)   # each branch leaves the stem at this angle (a taped "Y")


def _branch(x0, sep, a, side):
    """Bezier from (x0,0) leaving at FORK_ANGLE to side*sep, arriving parallel to x at x0+a."""
    k = 0.4 * a
    return bezier((x0, 0), (x0 + k, side * k * math.tan(FORK_ANGLE)), (x0 + a - k, side * sep),
                  (x0 + a, side * sep))


def eye_fork(x0, sep, length, side):
    """Branch leaving a stem at (x0,0), running at y=side*sep, rejoining at x0+length."""
    a = 0.35 * length
    p = _branch(x0, sep, a, side)
    q = seg_path((x0 + a, side * sep), 0.0, [("S", length - 2 * a)])
    r = _branch(x0 + length, sep, -a, side)[::-1]          # mirror image: merge back into the stem
    return join(p, q, r)


def open_fork(x0, sep, a, tail, side):
    """Branch leaving a stem at (x0,0) towards y=side*sep over a metres, then straight for tail metres."""
    return join(_branch(x0, sep, a, side), seg_path((x0 + a, side * sep), 0.0, [("S", tail)]))


def build_course(name, fork_plan):
    """Return a course dict: tapes (drawn), ghosts (metric-only centrelines), route (expected path
    for progress), start index on route, finish_s / finish zones, fork checkpoints, end point."""
    c = dict(name=name, tapes=[], ghosts=[], zones=[], checks=[], finish_s=None, tape_end=None,
             loop=False, start_s=0.5)
    if name in ("straight", "gap"):
        full = seg_path((-0.5, 0.0), 0.0, [("S", 6.5)])
        if name == "gap":
            xs = full[:, 0]
            c["tapes"] = [full[xs <= 2.5], full[xs >= 2.75]]
        else:
            c["tapes"] = [full]
        c["route"] = full
        c["finish_s"] = 5.5          # x = 5.0
    elif name == "end":
        tape = seg_path((-0.5, 0.0), 0.0, [("S", 3.5)])
        c["tapes"] = [tape]
        c["ghosts"] = [seg_path((3.0, 0.0), 0.0, [("S", 3.0)])]
        c["route"] = join(tape, c["ghosts"][0])
        c["tape_end"] = 3.5          # route arc length where the tape ends (x = 3.0)
    elif name == "s-curve":
        r = 0.7
        path = seg_path((-0.5, 0.0), 0.0, [("S", 1.5), ("A", r, math.pi / 2), ("A", r, -math.pi),
                                            ("A", r, math.pi / 2), ("S", 1.5)])
        c["tapes"] = [path]
        c["route"] = path
        c["finish_s"] = cumlen(path)[-1] - 0.5
    elif name == "oval":
        r, ls = 0.7, 2.0
        lap = seg_path((0.0, 0.0), 0.0, [("S", ls / 2), ("A", r, math.pi), ("S", ls), ("A", r, math.pi),
                                         ("S", ls / 2)])
        c["tapes"] = [lap]
        c["route"] = join(lap, lap)
        c["loop"] = True
        c["start_s"] = 0.0
        c["finish_s"] = cumlen(lap)[-1]
    elif name == "fork2":
        sep = 0.30
        plan = [s.strip().lower() for s in (fork_plan or "left,right").split(",") if s.strip()]
        while len(plan) < 2:
            plan.append("left")
        stem1 = seg_path((-0.5, 0.0), 0.0, [("S", 2.0)])                 # to x=1.5
        eyeL, eyeR = eye_fork(1.5, sep, 2.6, +1), eye_fork(1.5, sep, 2.6, -1)  # rejoin at x=4.1
        stem2 = seg_path((4.1, 0.0), 0.0, [("S", 0.9)])                  # to x=5.0
        endL, endR = open_fork(5.0, sep, 1.0, 1.0, +1), open_fork(5.0, sep, 1.0, 1.0, -1)  # end x=7.0
        c["tapes"] = [stem1, eyeL, eyeR, stem2, endL, endR]
        c["route"] = join(stem1, eyeL if plan[0] == "left" else eyeR, stem2,
                          endL if plan[1] == "left" else endR)
        c["zones"] = [("left", (6.6, sep), 0.2), ("right", (6.6, -sep), 0.2)]
        c["checks"] = [dict(name="fork1", expected=plan[0], pts={"left": (2.8, sep), "right": (2.8, -sep)}),
                       dict(name="fork2", expected=plan[1], pts={"left": (6.2, sep), "right": (6.2, -sep)})]
        c["expected_zone"] = plan[1]
        c["plan"] = ",".join(plan[:2])
    else:
        raise SystemExit("unknown course %s" % name)
    c["route_s"] = cumlen(c["route"])
    allpts = np.vstack(c["tapes"] + c["ghosts"])
    m = 1.2
    c["xmin"], c["xmax"] = allpts[:, 0].min() - m, allpts[:, 0].max() + m
    c["ymin"], c["ymax"] = allpts[:, 1].min() - m, allpts[:, 1].max() + m
    return c


# =============================================================================== world texture

def smooth_noise(rng, h, w, cell, amp):
    """Spatially correlated noise with feature size ~cell pixels."""
    sh, sw = max(2, h // cell + 2), max(2, w // cell + 2)
    n = rng.standard_normal((sh, sw)).astype(np.float32)
    return cv2.resize(n, (w, h), interpolation=cv2.INTER_CUBIC) * amp


def world_to_tex(c, x, y, ppm=PPM):
    return (x - c["xmin"]) * ppm, (c["ymax"] - y) * ppm


def build_world(c, rng, tape_w):
    W = int(math.ceil((c["xmax"] - c["xmin"]) * PPM))
    H = int(math.ceil((c["ymax"] - c["ymin"]) * PPM))
    # --- carpet: fine loop noise + streaks along x + low-frequency mottling, slightly green-grey
    fine = rng.normal(0.0, 22.0, (H, W)).astype(np.float32)
    fine = cv2.GaussianBlur(fine, (3, 3), 0.8)
    streak = rng.normal(0.0, 1.0, (H, W)).astype(np.float32)
    streak = cv2.blur(streak, (41, 1)) * 30.0
    mott = smooth_noise(rng, H, W, 120, 5.0)
    lum = fine + streak + mott
    tex = np.empty((H, W, 3), np.float32)
    for k, base in enumerate(CARPET_BGR):
        tex[:, :, k] = base + lum * (0.9 if k == 0 else 1.0)
    # --- tape mask with ragged edges
    mask = np.zeros((H, W), np.uint8)
    thick = max(1, int(round(tape_w * PPM)))
    for t in c["tapes"]:
        u, v = world_to_tex(c, t[:, 0], t[:, 1])
        pts = np.round(np.c_[u, v] * 16).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(mask, [pts], False, 255, thick, cv2.LINE_AA, shift=4)
    mf = cv2.GaussianBlur(mask.astype(np.float32) / 255.0, (0, 0), 1.5)
    rag = smooth_noise(rng, H, W, 6, 0.15) + smooth_noise(rng, H, W, 40, 0.06)
    alpha = np.clip((mf + rag - 0.5) * 4.0 + 0.5, 0.0, 1.0)
    alpha[mf < 0.02] = 0.0
    tnoise = rng.normal(0.0, 5.0, (H, W)).astype(np.float32) + smooth_noise(rng, H, W, 30, 3.0)
    for k, base in enumerate(TAPE_BGR):
        tex[:, :, k] = tex[:, :, k] * (1.0 - alpha) + (base + tnoise) * alpha
    tex = np.clip(tex, 0, 255).astype(np.uint8)
    # 4th channel = strength of the fine carpet detail added by the camera (paper tape is smoother)
    tex = np.dstack([tex, np.clip(255.0 * (1.0 - 0.75 * alpha), 0, 255).astype(np.uint8)])
    # --- lateral-error distance map (tape + ghost centrelines)
    Wr = int(math.ceil((c["xmax"] - c["xmin"]) * PPM_REF))
    Hr = int(math.ceil((c["ymax"] - c["ymin"]) * PPM_REF))
    ref = np.full((Hr, Wr), 255, np.uint8)
    for t in c["tapes"] + c["ghosts"]:
        u, v = world_to_tex(c, t[:, 0], t[:, 1], PPM_REF)
        pts = np.round(np.c_[u, v]).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(ref, [pts], False, 0, 1)
    dist = cv2.distanceTransform(ref, cv2.DIST_L2, 5) / PPM_REF
    return tex, dist


def lateral_error(c, dist, x, y):
    u, v = world_to_tex(c, x, y, PPM_REF)
    iu, iv = int(round(u)), int(round(v))
    if 0 <= iv < dist.shape[0] and 0 <= iu < dist.shape[1]:
        return float(dist[iv, iu])
    return float("inf")


# =============================================================================== camera

class Camera:
    """Pinhole camera on the car; per-pixel ground coordinates precomputed in the car frame."""

    def __init__(self, hfov_deg, height, horizon_y, cam_x, max_range, rng, detail=0.27):
        f = (W_IMG / 2.0) / math.tan(math.radians(hfov_deg) / 2.0)
        cx, cy = W_IMG / 2.0, H_IMG / 2.0
        th = math.atan((cy - horizon_y) / f)        # pitch down
        self.f, self.pitch, self.height = f, th, height
        u, v = np.meshgrid(np.arange(W_IMG, dtype=np.float64) + 0.5, np.arange(H_IMG, dtype=np.float64) + 0.5)
        xn, yn = (u - cx) / f, (v - cy) / f
        denom = yn * math.cos(th) + math.sin(th)    # ray z = -denom (camera looks down for denom > 0)
        ok = denom > 1e-6
        t = np.where(ok, height / np.where(ok, denom, 1.0), 0.0)
        X = t * (math.cos(th) - yn * math.sin(th))  # forward of the camera
        Y = -t * xn                                 # + = left
        self.valid = ok & (X < max_range) & (X > 0)
        self.Xc = np.where(self.valid, X + cam_x, 0.0).astype(np.float32)
        self.Yc = np.where(self.valid, Y, 0.0).astype(np.float32)
        self.bottom_dist = float(X[-1, W_IMG // 2])
        self.roi_top_dist = float(X[int(H_IMG * 0.5), W_IMG // 2])
        # lighting: near ground brighter, far darker, mild vignette (like the real frames)
        dist = np.where(self.valid, X, max_range)
        gain = 0.88 + 0.20 * np.clip((0.6 - dist) / 0.45, 0.0, 1.0)
        r2 = ((u - cx) / cx) ** 2 + ((v - cy) / cy) ** 2
        gain *= 1.0 - 0.08 * r2
        fade = np.clip((dist - (max_range - 1.0)) / 1.0, 0.0, 1.0)  # distant floor fades into haze
        validf = self.valid.astype(np.float32)
        # background above the horizon / beyond max range: dim wall with some vertical structure
        col = smooth_noise(rng, 1, W_IMG, 25, 12.0)
        rowg = np.linspace(-10.0, 10.0, H_IMG, dtype=np.float32)[:, None]
        base = 92.0 + col + rowg + rng.normal(0, 4.0, (H_IMG, W_IMG)).astype(np.float32)
        haze = (75.0, 102.0, 100.0)
        # per frame: img = ground * A * (1 + detail) + B
        self.A = (gain * (1.0 - fade) * validf).astype(np.float32)
        self.B = cv2.merge([(haze[k] * fade * validf + base * m * (1.0 - validf)).astype(np.float32)
                            for k, m in enumerate((0.78, 1.05, 1.0))])
        # fine carpet detail (loops ~2-3 mm) as a small wrapped tile at 1 mm/px, so the magnified near
        # floor keeps real-looking texture (the 4 px/cm ground texture alone looks blurred up close)
        def wrap_filter(fn, arr):      # filter as if the tile repeats (OpenCV filters have no WRAP border)
            return fn(np.tile(arr, (3, 3)))[256:512, 256:512]
        tile = wrap_filter(lambda m: cv2.GaussianBlur(m, (0, 0), 0.9),
                           rng.normal(0.0, 1.0, (256, 256)).astype(np.float32))
        tile = tile + 0.6 * wrap_filter(lambda m: cv2.blur(m, (9, 1)),
                                        rng.normal(0.0, 1.0, (256, 256)).astype(np.float32))
        tile *= detail / float(tile.std())
        self.tile = np.ascontiguousarray(tile, dtype=np.float32)
        self.noise_sigma, self.noise_bank = None, []

    def render(self, c, tex, pose, rng, dark=1.0, noise=3.0):
        x, y, yaw = pose
        cs, sn = math.cos(yaw), math.sin(yaw)
        mx = cv2.addWeighted(self.Xc, cs * PPM, self.Yc, -sn * PPM, (x - c["xmin"]) * PPM)
        my = cv2.addWeighted(self.Xc, -sn * PPM, self.Yc, -cs * PPM, (c["ymax"] - y) * PPM)
        ground = cv2.remap(tex, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101).astype(np.float32)
        dm = DETAIL_PPM / PPM
        d = cv2.remap(self.tile, mx * dm, my * dm, cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
        fac = cv2.multiply(self.A, cv2.multiply(d, ground[:, :, 3], scale=1.0 / 255.0) + 1.0)
        img = cv2.multiply(ground[:, :, :3], cv2.merge([fac, fac, fac]))
        img = cv2.add(img, self.B)
        img = cv2.GaussianBlur(img, (0, 0), 0.6)
        if dark != 1.0:
            img *= dark
        if noise > 0:
            if self.noise_sigma != noise:
                self.noise_sigma = noise
                self.noise_bank = []
                for _ in range(6):
                    n = np.empty((H_IMG, W_IMG * 3), np.float32)
                    cv2.randn(n, 0.0, noise)
                    self.noise_bank.append(n.reshape(H_IMG, W_IMG, 3))
            n = self.noise_bank[int(rng.integers(len(self.noise_bank)))]
            sh = int(rng.integers(0, 64))         # cheap decorrelation between frames
            img[:, sh:] += n[:, :W_IMG - sh]
            img[:, :sh] += n[:, W_IMG - sh:]
        np.clip(img, 0, 255, out=img)
        return img.astype(np.uint8)


# =============================================================================== vehicle

class Vehicle:
    """Kinematic bicycle (reference point = rear axle). steer +1 = LEFT."""

    def __init__(self, a, pose):
        self.x, self.y, self.yaw = pose
        self.v = 0.0
        self.delta = 0.0
        self.L = a.wheelbase
        self.max_delta = a.max_steer_angle
        self.vmax, self.dead, self.tau = a.vmax, a.dead, a.tau
        self.dance = a.esc_reverse_dance
        self.rev_armed = True
        self.neutral_t = 0.0
        self.odo = 0.0

    def speed_for(self, thr):
        mag = max(0.0, abs(thr) - self.dead) / (1.0 - self.dead)
        return math.copysign(self.vmax * min(1.0, mag), thr) if mag > 0 else 0.0

    def update(self, thr, steer, dt):
        tau = self.tau
        target = self.speed_for(thr)
        if self.dance:
            # hobby ESC: after forward drive, a negative command brakes; reverse only engages once the
            # throttle has rested in neutral (|thr| <= 0.05) for >= 0.1 s
            if thr > 0.05:
                self.rev_armed, self.neutral_t = False, 0.0
            elif thr >= -0.05:
                self.neutral_t += dt
                if self.neutral_t >= 0.1:
                    self.rev_armed = True
            else:
                self.neutral_t = 0.0
                if not self.rev_armed:
                    target, tau = 0.0, 0.08   # brake
        if thr < -0.05 and self.v > 0.02 and not self.dance:
            tau = 0.1
        self.v += (target - self.v) * min(1.0, dt / tau)
        d_cmd = max(-1.0, min(1.0, steer)) * self.max_delta
        self.delta += (d_cmd - self.delta) * min(1.0, dt / 0.06)
        self.x += self.v * math.cos(self.yaw) * dt
        self.y += self.v * math.sin(self.yaw) * dt
        self.yaw += self.v / self.L * math.tan(self.delta) * dt
        self.odo += abs(self.v) * dt

    @property
    def pose(self):
        return (self.x, self.y, self.yaw)

    def centre(self):
        h = 0.5 * self.L
        return self.x + h * math.cos(self.yaw), self.y + h * math.sin(self.yaw)


# =============================================================================== control glue

class LegacyController:
    """Adapter: old stateless auto_node.compute() behind the AutoController API."""

    def __init__(self, p):
        self.p = p
        self.state = "tracking"

    def step(self, now, offset, view_ok, offset_age, ok_age, confidence=None, mode="auto"):
        if mode != "auto":
            return dict(throttle=0.0, steer=0.0, reason="mode=%s" % mode, state="idle", request=None)
        thr, steer, reason = auto_node.compute(offset, view_ok, offset_age, ok_age, self.p)
        self.state = reason
        return dict(throttle=thr, steer=steer, reason=reason, state=reason, request=None)


class MiniArbiter:
    """Fallback if control_node.Arbiter is unavailable: STOP on view_ok False/stale (>1 s), watchdog,
    and stop requests. Same interface subset as control_node.Arbiter."""
    VIEW_TIMEOUT_S, CMD_TIMEOUT_S = 1.0, 0.5

    def __init__(self, log_fn):
        self.log, self.mode, self.stop_reason = log_fn, "stop", "startup"
        self.auto = self.manual = self.view = None

    def on_manual(self, t, s, now):
        self.manual = (t, s, now)

    def on_auto(self, t, s, now):
        self.auto = (t, s, now)

    def on_view(self, ok, now):
        self.view = (bool(ok), now)

    def view_good(self, now):
        return self.view is not None and self.view[0] and now - self.view[1] <= self.VIEW_TIMEOUT_S

    def _set(self, m, why):
        if m != self.mode:
            self.log("MODE_CHANGE", self.mode, m, why)
            self.mode = m
            if m == "stop":
                self.stop_reason = why

    def request(self, raw, now):
        req = raw.split(":")[0].strip().lower()
        if req == "stop":
            self._set("stop", "request " + raw)
        elif req == "resume" and self.mode == "stop":
            self._set("manual", "resume")
        elif req == "auto" and self.mode != "stop" and self.auto and now - self.auto[2] <= 0.5 \
                and self.view_good(now):
            self._set("auto", "human request (auto)")

    def step(self, now):
        if self.mode == "auto":
            if now - self.auto[2] > self.CMD_TIMEOUT_S:
                self._set("stop", "WATCHDOG")
            elif not self.view_good(now):
                self._set("stop", "PERCEPTION: view_ok bad")
        if self.mode == "stop":
            return 0.0, 0.0
        s = self.auto if self.mode == "auto" else self.manual
        return max(-1.0, min(1.0, s[0])), max(-1.0, min(1.0, s[1]))


def make_controller(auto_argv):
    pa = auto_node.parse_args(auto_argv)
    if hasattr(auto_node, "AutoController"):
        ctl = auto_node.AutoController(pa)
        kind = "AutoController"
    else:
        ctl = LegacyController(pa)
        kind = "compute() adapter"
    try:
        params = inspect.signature(ctl.step).parameters
    except (TypeError, ValueError):
        params = {}
    return ctl, kind, pa, params


# =============================================================================== map panel

class MapPanel:
    def __init__(self, c, tex, w=640, h=400):
        cw, ch = c["xmax"] - c["xmin"], c["ymax"] - c["ymin"]
        self.s = min(w / cw, h / ch)          # px per metre
        self.w, self.h = w, h
        self.ox = (w - cw * self.s) / 2.0
        self.oy = (h - ch * self.s) / 2.0
        small = cv2.resize(tex[:, :, :3], (int(cw * self.s), int(ch * self.s)), interpolation=cv2.INTER_AREA)
        self.base = np.zeros((h, w, 3), np.uint8)
        y0, x0 = int(self.oy), int(self.ox)
        self.base[y0:y0 + small.shape[0], x0:x0 + small.shape[1]] = small
        self.c = c
        for g in c["ghosts"]:
            pts = self.pts(g)
            for i in range(0, len(pts) - 8, 16):
                cv2.line(self.base, tuple(pts[i]), tuple(pts[i + 8]), (120, 120, 120), 1)
        for name, (zx, zy), r in c["zones"]:
            cv2.circle(self.base, self.p(zx, zy), max(2, int(r * self.s)), (0, 200, 255), 1)
        if c["finish_s"] is not None and not c["loop"]:
            i = int(np.searchsorted(c["route_s"], c["finish_s"]))
            fx, fy = c["route"][min(i, len(c["route"]) - 1)]
            cv2.drawMarker(self.base, self.p(fx, fy), (0, 200, 255), cv2.MARKER_TILTED_CROSS, 12, 2)

    def p(self, x, y):
        return (int(self.ox + (x - self.c["xmin"]) * self.s), int(self.oy + (self.c["ymax"] - y) * self.s))

    def pts(self, arr):
        return np.c_[self.ox + (arr[:, 0] - self.c["xmin"]) * self.s,
                     self.oy + (self.c["ymax"] - arr[:, 1]) * self.s].astype(np.int32)

    def draw(self, trail, veh, cam):
        img = self.base.copy()
        if len(trail) > 1:
            cv2.polylines(img, [self.pts(np.asarray(trail)).reshape(-1, 1, 2)], False, (255, 80, 255), 1)
        x, y, yaw = veh.pose
        cs, sn = math.cos(yaw), math.sin(yaw)
        # field of view on the floor (ROI bottom half: roi_top_dist .. bottom_dist)
        hf = math.tan(math.atan((W_IMG / 2.0) / cam.f))
        fov = []
        for d, side in ((cam.bottom_dist, 1), (cam.roi_top_dist, 1), (cam.roi_top_dist, -1),
                        (cam.bottom_dist, -1)):
            slant = math.hypot(d, cam.height)
            lx, ly = d + veh_cam_x(veh), side * slant * hf
            fov.append(self.p(x + lx * cs - ly * sn, y + lx * sn + ly * cs))
        cv2.polylines(img, [np.array(fov, np.int32)], True, (0, 255, 255), 1)
        body = []
        for lx, ly in ((-0.05, 0.09), (0.25, 0.09), (0.25, -0.09), (-0.05, -0.09)):
            body.append(self.p(x + lx * cs - ly * sn, y + lx * sn + ly * cs))
        cv2.polylines(img, [np.array(body, np.int32)], True, (0, 0, 255), 2)
        cv2.circle(img, self.p(x, y), 2, (0, 0, 255), -1)
        return img


_CAM_X = [0.12]


def veh_cam_x(_veh):
    return _CAM_X[0]


# =============================================================================== run one course

def run_course(name, a, out_dir):
    rng = np.random.default_rng(a.seed)
    c = build_course(name, a.fork_plan)
    tex, dist = build_world(c, rng, a.tape_width)
    cam = Camera(a.hfov, a.cam_height, a.horizon_y, a.cam_x, a.max_range, rng)
    _CAM_X[0] = a.cam_x

    # --- perception (real camera_node)
    cam_argv = shlex.split(a.cam_args or "")
    probe = camera_node.parse_args([])
    plan_used = None
    if name == "fork2" or a.fork_plan:
        plan = c.get("plan", a.fork_plan)
        if hasattr(probe, "fork_plan"):
            if plan:
                cam_argv += ["--fork-plan", plan]
                plan_used = plan
        elif a.fork_plan or name == "fork2":
            print("  [warn] camera_node has no --fork-plan; fork plan '%s' NOT passed to perception" % plan)
    pp = camera_node.parse_args(cam_argv)
    per = camera_node.Perception(pp, single_shot=False)

    # --- controller (real auto_node) + arbiter (real control_node)
    auto_argv = ["--throttle", str(a.throttle)] + shlex.split(a.auto_args or "")
    ctl, ctl_kind, pa, ctl_params = make_controller(auto_argv)
    events = []

    def arb_log(event, frm, to, reason):
        events.append((now_ref[0], event, frm, to, reason))

    now_ref = [0.0]
    arb = control_node.Arbiter(arb_log) if HAVE_ARBITER else MiniArbiter(arb_log)
    arb_kind = "control_node.Arbiter" if HAVE_ARBITER else "MiniArbiter"
    ctl_every = max(1, int(round(SIM_HZ / pa.rate)))

    # --- start pose
    i0 = int(np.searchsorted(c["route_s"], c["start_s"]))
    sx, sy = c["route"][i0]
    h0 = heading_at(c["route"], i0)
    sx, sy = sx - a.start_offset * math.sin(h0), sy + a.start_offset * math.cos(h0)   # + = left of tape
    veh = Vehicle(a, (sx, sy, h0 + math.radians(a.start_yaw)))
    prog_i = i0

    # --- video
    writer, png_dir, vid_path = None, None, ""
    if not a.no_video:
        vid_path = os.path.join(out_dir, "%s.mp4" % name)
        fps = max(1.0, 15.0 / a.video_every)
        writer = cv2.VideoWriter(vid_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W_IMG * 2, H_IMG))
        if not writer.isOpened():
            writer = None
            png_dir = os.path.join(out_dir, "%s_frames" % name)
            os.makedirs(png_dir, exist_ok=True)
            vid_path = png_dir
            print("  [warn] mp4v VideoWriter failed; writing PNG frames to %s" % png_dir)
        mp = MapPanel(c, tex)

    trace_f = open(os.path.join(out_dir, "%s_trace.csv" % name), "w", newline="")
    trace = csv.writer(trace_f)
    trace.writerow(["t", "x", "y", "yaw", "v", "mode", "state", "reason", "thr_cmd", "steer_cmd",
                    "thr_out", "steer_out", "offset", "view_ok", "conf", "lat_cm", "progress_m"])

    dt = 1.0 / SIM_HZ
    pending = []                 # (publish_time, shown, info, det)
    last_info, last_det, last_shown = None, None, None
    last_msg_t = None
    cmd = dict(throttle=0.0, steer=0.0, reason="starting", state="-", request=None)
    out_thr = out_steer = 0.0
    auto_t0 = None
    stop_t = None
    still_t = 0.0
    trail = []
    lats = []
    max_lat = 0.0
    reverses = 0
    prev_state = None
    choices = {}
    outcome, stop_reason = "timeout", ""
    frame_n = 0
    per_ms = []
    perception_frames = 0
    nan_frames = 0
    t_wall = time.perf_counter()
    nsteps = int(a.duration * SIM_HZ)
    resumed = False
    fork_events = []

    for k in range(nsteps):
        now = k * dt
        now_ref[0] = now

        # ---------------- camera capture (15 Hz) -> processed result published one frame later
        if k % CAM_EVERY == 0:
            img = cam.render(c, tex, veh.pose, rng, a.dark, a.noise)
            tp = time.perf_counter()
            shown, info, det = per.step(img, None, a.degrade)
            per_ms.append((time.perf_counter() - tp) * 1000.0)
            pending.append((now + CAM_EVERY * dt - 1e-9, shown, info, det))
        while pending and pending[0][0] <= now:
            _, last_shown, last_info, last_det = pending.pop(0)
            last_msg_t = now
            arb.on_view(bool(last_info["view_ok"]), now)
            perception_frames += 1
            off = last_info["offset"]
            if off is None or not math.isfinite(off):
                nan_frames += 1
            if last_info.get("fork"):
                fork_events.append((round(now, 2), last_info.get("fork_side"), last_info.get("fork_count")))

        # ---------------- human: teleop sends zeros; press R at start, O once the arbiter will accept
        arb.on_manual(0.0, 0.0, now)
        if not resumed:
            arb.request("resume", now)
            resumed = True

        # ---------------- controller (20 Hz)
        if k % ctl_every == 0:
            if last_info is None:
                off, ok, conf, age = float("nan"), False, None, float("inf")
            else:
                off = last_info["offset"]
                off = float("nan") if off is None else float(off)
                ok, conf, age = bool(last_info["view_ok"]), last_info.get("confidence"), now - last_msg_t
            kwargs = {}
            if "confidence" in ctl_params:
                kwargs["confidence"] = conf
            if "mode" in ctl_params:
                kwargs["mode"] = arb.mode
            for key in ("fork", "fork_side", "fork_count"):
                if key in ctl_params and last_info is not None:
                    kwargs[key] = last_info.get(key)
            cmd = ctl.step(now, off, ok, age, age, **kwargs)
            arb.on_auto(cmd["throttle"], cmd["steer"], now)
            st = cmd.get("state")
            if st is not None and st != prev_state:
                if "rev" in str(st).lower() and not (prev_state and "rev" in str(prev_state).lower()):
                    reverses += 1
                prev_state = st
            req = cmd.get("request")
            if req and str(req).lower().startswith("stop"):
                arb.request(str(req), now)

        # the human presses O once the dashboard shows view_ok True (avoids a stream of REJECTED logs)
        if arb.mode == "manual" and auto_t0 is None and (not hasattr(arb, "view_good") or arb.view_good(now)):
            arb.request("auto", now)
            if arb.mode == "auto":
                auto_t0 = now

        # ---------------- arbiter (every sim step) + vehicle
        out_thr, out_steer = arb.step(now)
        veh.update(out_thr, out_steer, dt)

        # ---------------- metrics
        cxm, cym = veh.centre()
        lat = lateral_error(c, dist, cxm, cym)
        lo, hi = max(0, prog_i - 100), min(len(c["route"]), prog_i + 400)
        d2 = (c["route"][lo:hi, 0] - cxm) ** 2 + (c["route"][lo:hi, 1] - cym) ** 2
        prog_i = lo + int(np.argmin(d2))
        progress = c["route_s"][prog_i] - c["start_s"]
        if k % 2 == 0:
            trail.append((veh.x, veh.y))
        for chk in c["checks"]:
            if chk["name"] in choices:
                continue
            for side, (px, py) in chk["pts"].items():
                if math.hypot(cxm - px, cym - py) < 0.15:
                    choices[chk["name"]] = side
        if k % 6 == 0:
            off = last_info["offset"] if last_info else float("nan")
            trace.writerow(["%.3f" % now, "%.4f" % veh.x, "%.4f" % veh.y, "%.4f" % veh.yaw, "%.3f" % veh.v,
                            arb.mode, cmd.get("state"), cmd.get("reason"), "%.3f" % cmd["throttle"],
                            "%.3f" % cmd["steer"], "%.3f" % out_thr, "%.3f" % out_steer,
                            "" if off is None or not math.isfinite(off) else "%.3f" % off,
                            last_info["view_ok"] if last_info else "",
                            "%.2f" % last_info["confidence"] if last_info else "",
                            "%.1f" % (lat * 100 if math.isfinite(lat) else 999), "%.3f" % progress])

        # ---------------- video
        if (writer is not None or png_dir) and k % CAM_EVERY == 0 and (k // CAM_EVERY) % a.video_every == 0:
            if last_shown is not None:
                left = camera_node.draw_debug(last_shown, last_info, last_det)
                fk = [("fork=%s side=%s n=%s" % (last_info.get("fork"), last_info.get("fork_side"),
                                                 last_info.get("fork_count")))] if "fork" in last_info else []
                for j, txt in enumerate(fk):
                    cv2.putText(left, txt, (8, 470 - 22 * j), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1,
                                cv2.LINE_AA)
            else:
                left = np.zeros((H_IMG, W_IMG, 3), np.uint8)
            right = np.zeros((H_IMG, W_IMG, 3), np.uint8)
            right[:400] = mp.draw(trail, veh, cam)
            lines = [
                "%s  t=%.2fs  mode=%s  ctl=%s" % (name, now, arb.mode.upper(), cmd.get("state")),
                "thr cmd %+.2f  out %+.2f   steer cmd %+.2f  out %+.2f  v=%.2f m/s"
                % (cmd["throttle"], out_thr, cmd["steer"], out_steer, veh.v),
                "reason: %s" % str(cmd.get("reason"))[:60],
                "lat %.1f cm  progress %.2f m  reverses %d%s" % (
                    lat * 100 if math.isfinite(lat) else 999, progress, reverses,
                    ("  stop: " + arb.stop_reason[:28]) if arb.mode == "stop" and auto_t0 is not None else ""),
            ]
            for j, txt in enumerate(lines):
                cv2.putText(right, txt, (8, 418 + 18 * j), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1,
                            cv2.LINE_AA)
            frame = np.hstack([left, right])
            if writer is not None:
                writer.write(frame)
            else:
                cv2.imwrite(os.path.join(png_dir, "f%05d.png" % frame_n), frame)
            frame_n += 1

        # ---------------- end conditions
        if auto_t0 is None:
            if now > 5.0:
                outcome, stop_reason = "no-start", "arbiter never accepted AUTO (view_ok=%s)" % (
                    last_info["view_ok"] if last_info else None)
                break
            continue
        if math.isfinite(lat) is False or lat > 0.25:
            outcome, stop_reason = "left-course", "lateral error %.0f cm; ctl: %s" % (
                (lat * 100 if math.isfinite(lat) else 999), cmd.get("reason"))
            break
        done_zone = None
        for zname, (zx, zy), r in c["zones"]:
            if math.hypot(cxm - zx, cym - zy) < r:
                done_zone = zname
        if done_zone or (c["finish_s"] is not None and progress >= c["finish_s"] - (c["start_s"] if not c["loop"]
                                                                                 else 0.0)):
            outcome = "finished"
            stop_reason = "end zone %s" % done_zone if done_zone else "end zone"
            if done_zone:
                choices.setdefault("zone", done_zone)
            break
        if arb.mode == "stop":
            if stop_t is None:
                stop_t = now
            if abs(veh.v) < 0.02 and now - stop_t > 0.3:
                outcome, stop_reason = "stopped", arb.stop_reason
                break
        moving = abs(veh.v) > 0.01
        still_t = 0.0 if moving else still_t + dt
        if arb.mode == "auto" and still_t >= 3.0:
            outcome, stop_reason = "gave-up", "no motion for 3 s in AUTO; ctl: %s" % cmd.get("reason")
            break
        if math.isfinite(lat):
            lats.append(lat)
            max_lat = max(max_lat, lat)

    if outcome == "timeout":
        stop_reason = "duration %.0fs reached; ctl: %s" % (a.duration, cmd.get("reason"))
    trace_f.close()
    if writer is not None:
        writer.release()

    # ---------------- pass/fail
    cxm, cym = veh.centre()
    ok = False
    note = ""
    taken = ",".join(choices.get(k, "-") for k in ("fork1", "fork2")) if c["checks"] else ""
    expected = c.get("plan", "")
    if name == "end":
        over = c["route_s"][prog_i] - c["tape_end"]
        note = "overrun %.2f m past tape end" % over
        ok = outcome in ("stopped", "gave-up") and over <= 0.75
    elif name == "fork2":
        ok = outcome == "finished" and taken == expected and choices.get("zone") == c["expected_zone"]
    else:
        ok = outcome == "finished"
    t_end = now
    res = dict(course=name, outcome=outcome, passed="PASS" if ok else "FAIL",
               time_s=round(t_end - (auto_t0 or 0.0), 2),
               dist_m=round(c["route_s"][prog_i] - c["start_s"], 2), odo_m=round(veh.odo, 2),
               max_lat_cm=round(max_lat * 100, 1),
               mean_lat_cm=round(float(np.mean(lats)) * 100 if lats else 0.0, 1),
               reverses=reverses, forks_taken=taken, forks_expected=expected,
               stop_reason=(stop_reason + ("; " + note if note else "")).replace("\n", " "),
               nan_frac=round(nan_frames / max(1, perception_frames), 2),
               per_ms=round(float(np.mean(per_ms)) if per_ms else 0.0, 1),
               wall_s=round(time.perf_counter() - t_wall, 1), controller=ctl_kind, arbiter=arb_kind,
               fork_plan_to_cam=plan_used or "", video=vid_path)
    res["_events"] = events
    res["_fork_events"] = fork_events
    return res


# =============================================================================== main

def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="Offline Jafara line-following simulator (real camera_node / "
                                             "auto_node / control_node logic, synthetic camera)")
    ap.add_argument("--course", default="all", choices=COURSES + ["all"])
    ap.add_argument("--throttle", type=float, default=0.5, help="passed to auto_node --throttle (default 0.5)")
    ap.add_argument("--auto-args", default="", help='extra args for auto_node.parse_args, e.g. "--k 1.2"')
    ap.add_argument("--cam-args", default="", help='extra args for camera_node.parse_args, e.g. "--z-thresh 2.5"')
    ap.add_argument("--dark", type=float, default=1.0, help="brightness factor on the rendered colour image")
    ap.add_argument("--noise", type=float, default=3.0, help="sensor noise sigma (gray levels)")
    ap.add_argument("--degrade", type=int, default=0, help="blur steps passed to Perception.step (0..10)")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--start-offset", type=float, default=0.03,
                    help="start this far (m) left of the tape centre (+ = left), default 0.03")
    ap.add_argument("--start-yaw", type=float, default=-4.0,
                    help="start heading error in degrees (+ = pointing left), default -4")
    ap.add_argument("--duration", type=float, default=60.0, help="max sim seconds per course")
    ap.add_argument("--fork-plan", default=None,
                    help='e.g. "left,right" (fork2 default); passed to camera_node if it has --fork-plan')
    ap.add_argument("--vmax", type=float, default=1.2, help="m/s at full throttle")
    ap.add_argument("--dead", type=float, default=0.3, help="throttle deadband (|thr| below this = no motion)")
    ap.add_argument("--tau", type=float, default=0.2, help="speed lag time constant (s)")
    ap.add_argument("--esc-reverse-dance", action="store_true",
                    help="model the ESC: reverse only after >=0.1 s of neutral following forward drive")
    ap.add_argument("--wheelbase", type=float, default=0.16)
    ap.add_argument("--max-steer-angle", type=float, default=0.45, help="rad at steer = +-1")
    ap.add_argument("--tape-width", type=float, default=0.05, help="m")
    ap.add_argument("--hfov", type=float, default=69.0, help="camera horizontal FOV (deg)")
    ap.add_argument("--cam-height", type=float, default=0.12, help="camera height above floor (m)")
    ap.add_argument("--horizon-y", type=float, default=100.0, help="image row of the horizon (sets the pitch)")
    ap.add_argument("--cam-x", type=float, default=0.12, help="camera position ahead of the rear axle (m)")
    ap.add_argument("--max-range", type=float, default=3.0, help="floor rendered up to this distance (m)")
    ap.add_argument("--no-video", action="store_true")
    ap.add_argument("--video-every", type=int, default=1, help="write every Nth camera frame (default 1)")
    ap.add_argument("--out", default=None, help="output dir (default ./sim_out/<timestamp>/ next to sim.py)")
    a = ap.parse_args(argv)
    if not 0.0 <= a.dead < 1.0:
        ap.error("--dead must be 0..<1")
    a.video_every = max(1, a.video_every)
    a.degrade = max(0, min(camera_node.DEGRADE_MAX, a.degrade))
    return a


def print_table(rows):
    cols = ["course", "outcome", "passed", "time_s", "dist_m", "max_lat_cm", "mean_lat_cm", "reverses",
            "forks_taken", "forks_expected", "stop_reason"]
    width = {k: max(len(k), *(len(str(r[k])) for r in rows)) for k in cols}
    width["stop_reason"] = min(width["stop_reason"], 70)
    print(" | ".join(k.ljust(width[k]) for k in cols))
    print("-+-".join("-" * width[k] for k in cols))
    for r in rows:
        print(" | ".join(str(r[k])[:width[k]].ljust(width[k]) for k in cols))


def main(argv=None):
    a = parse_args(argv)
    out_dir = a.out or os.path.join(SCRIPT_DIR, "sim_out", datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(out_dir, exist_ok=True)
    courses = COURSES if a.course == "all" else [a.course]
    cam = Camera(a.hfov, a.cam_height, a.horizon_y, a.cam_x, a.max_range, np.random.default_rng(0))
    print("camera: f=%.0f px pitch=%.1f deg height=%.2f m; image bottom = %.2f m ahead of the camera, "
          "ROI top (y=240) = %.2f m" % (cam.f, math.degrees(cam.pitch), a.cam_height, cam.bottom_dist,
                                        cam.roi_top_dist))
    print("controller: %s   arbiter: %s   output: %s" % (
        "auto_node.AutoController" if hasattr(auto_node, "AutoController") else "auto_node.compute() adapter",
        "control_node.Arbiter" if HAVE_ARBITER else "MiniArbiter (fallback)", out_dir))
    rows = []
    for name in courses:
        print("[%s] running ..." % name, flush=True)
        r = run_course(name, a, out_dir)
        print("[%s] %s %s in %.1fs sim (%.1fs wall, perception %.1f ms/frame) - %s" % (
            name, r["outcome"], r["passed"], r["time_s"], r["wall_s"], r["per_ms"], r["stop_reason"]), flush=True)
        for ev in r["_events"]:
            print("      t=%6.2f %-12s %s -> %s : %s" % ev)
        if r["_fork_events"]:
            print("      fork detections (t, side, count): %s%s" % (
                r["_fork_events"][:12], " ..." if len(r["_fork_events"]) > 12 else ""))
        rows.append(r)
    keys = [k for k in rows[0] if not k.startswith("_")]
    with open(os.path.join(out_dir, "results.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print()
    print_table(rows)
    print("\nresults: %s" % os.path.join(out_dir, "results.csv"))
    return 0 if all(r["passed"] == "PASS" for r in rows) else 1


if __name__ == "__main__":
    sys.exit(main())
