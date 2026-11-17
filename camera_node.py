#!/usr/bin/env python3
"""
FloodBot perception node (Person A).

Contract (do not rename):
  /perception/line_offset  std_msgs/Float32  -1.0 far left .. 0.0 center .. +1.0 far right, NaN = no line
  /perception/view_ok      std_msgs/Bool     True = image usable, False = too blurred / dark / occluded

Extra (for the dashboard, not part of the contract):
  /perception/debug_image/compressed  sensor_msgs/CompressedImage  annotated JPEG (tape outline,
      offset, view_ok). Only encoded while something subscribes. --image-hz 0 turns it off.

Modes:
  python3 camera_node.py                 # ROS 2 node, ~15 Hz (source /opt/ros/humble/setup.bash first)
  python3 camera_node.py --test          # no ROS: grab one frame, print offset + view_ok, save debug.jpg
  python3 camera_node.py --test --image some.jpg   # no ROS, no camera: run detection on a saved image

Useful options (work in every mode):
  --line dark|light      dark tape on light floor (default) or light tape on dark floor
  --thresh N             gray-level threshold for the tape (default 90 for dark, 170 for light; -1 = Otsu auto)
  --blur-min N           Laplacian-variance floor; below this view_ok=False
  --bright-min/--bright-max N   mean-brightness window; outside it view_ok=False
  --degrade N            artificially blur the frame by N steps (for the G2 poor-view test)
  --ramp SECS            (ROS mode) step degrade 0,1,..,N every SECS seconds, then repeat
"""

import argparse
import array
import math
import os
import sys
import time

import cv2
import numpy as np

DEFAULT_DEBUG_PATH = os.path.expanduser("~/robo-cam/debug.jpg")
WIDTH, HEIGHT, FPS = 640, 480, 15


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
    def __init__(self, width=WIDTH, height=HEIGHT, fps=FPS):
        rs = import_realsense()
        self.rs = rs
        self.pipeline = rs.pipeline()
        cfg = rs.config()
        cfg.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
        try:
            self.pipeline.start(cfg)
        except RuntimeError as e:
            sys.stderr.write(
                "[camera_node] Failed to start RealSense color stream: %s\n"
                "  - Is another process using the camera (realsense-viewer, realsense2_camera node)?\n"
                "  - Is the D435 on a USB3 (blue) port? Check: rs-enumerate-devices -s\n" % e
            )
            raise SystemExit(1)

    def warmup(self, n=30):
        """Drop frames so auto-exposure settles (important for single-shot test mode)."""
        for _ in range(n):
            self.pipeline.wait_for_frames(5000)

    def read(self, timeout_ms=1000):
        frames = self.pipeline.wait_for_frames(timeout_ms)
        color = frames.get_color_frame()
        if not color:
            return None
        return np.asanyarray(color.get_data()).copy()

    def stop(self):
        try:
            self.pipeline.stop()
        except Exception:
            pass


# --------------------------------------------------------------------------- vision

def degrade(img, steps):
    """Artificial blur for G2. Each step widens the Gaussian kernel by 4 px (step 1 = 5x5, 5 = 21x21)."""
    if steps <= 0:
        return img
    k = 4 * steps + 1
    return cv2.GaussianBlur(img, (k, k), 0)


def check_view(gray, p):
    """Return (view_ok, laplacian_variance, mean_brightness, reason)."""
    lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    bright = float(gray.mean())
    reasons = []
    if lap_var < p.blur_min:
        reasons.append("blurry")
    if bright < p.bright_min:
        reasons.append("dark")
    if bright > p.bright_max:
        reasons.append("washed-out")
    return (not reasons), lap_var, bright, ",".join(reasons) or "ok"


def find_line(gray, p):
    """
    Look for the tape in the bottom ROI of the image.
    Returns (offset or NaN, contour or None, (cx, cy) or None, roi_top_px, threshold_used).
    """
    h, w = gray.shape
    top = int(h * p.roi_top)
    roi = cv2.GaussianBlur(gray[top:, :], (5, 5), 0)

    want_dark = p.line == "dark"
    mode = cv2.THRESH_BINARY_INV if want_dark else cv2.THRESH_BINARY
    if p.thresh < 0:
        t, mask = cv2.threshold(roi, 0, 255, mode | cv2.THRESH_OTSU)
    else:
        t, mask = cv2.threshold(roi, p.thresh, 255, mode)

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    roi_area = roi.shape[0] * roi.shape[1]
    best, best_area = None, 0.0
    for c in contours:
        a = cv2.contourArea(c)
        # too small = noise; too big = the threshold grabbed the floor/shadow, not tape
        if a < p.min_area or a > p.max_area_frac * roi_area:
            continue
        if a > best_area:
            best, best_area = c, a
    if best is None:
        return float("nan"), None, None, top, t

    m = cv2.moments(best)
    cx = m["m10"] / m["m00"]
    cy = m["m01"] / m["m00"] + top
    offset = (cx - w / 2.0) / (w / 2.0)
    offset = max(-1.0, min(1.0, offset))
    best = best + np.array([[[0, top]]], dtype=best.dtype)  # shift to full-image coords
    return offset, best, (int(cx), int(cy)), top, t


def process(frame, p, degrade_steps):
    frame = degrade(frame, degrade_steps)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    view_ok, lap_var, bright, reason = check_view(gray, p)
    offset, contour, centroid, top, t = find_line(gray, p)
    info = dict(offset=offset, view_ok=view_ok, lap_var=lap_var, bright=bright,
                reason=reason, thresh=t, degrade=degrade_steps)
    return frame, info, contour, centroid, top


def draw_debug(frame, info, contour, centroid, top):
    img = frame.copy()
    h, w = img.shape[:2]
    cv2.rectangle(img, (0, top), (w - 1, h - 1), (255, 200, 0), 1)          # ROI
    cv2.line(img, (w // 2, top), (w // 2, h), (200, 200, 200), 1)           # image center
    if contour is not None:
        cv2.drawContours(img, [contour], -1, (0, 255, 0), 2)
        cv2.circle(img, centroid, 7, (0, 0, 255), -1)
        cv2.line(img, (w // 2, centroid[1]), centroid, (0, 0, 255), 2)
    off = info["offset"]
    off_s = "NaN (no line)" if math.isnan(off) else "%+.2f" % off
    ok_col = (0, 200, 0) if info["view_ok"] else (0, 0, 255)
    lines = [
        ("offset: %s" % off_s, (255, 255, 255)),
        ("view_ok: %s (%s)" % (info["view_ok"], info["reason"]), ok_col),
        ("lapvar: %.0f  bright: %.0f  thr: %.0f  degrade: %d"
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
    return "offset=%s view_ok=%s (%s) lapvar=%.1f bright=%.1f thr=%.0f degrade=%d" % (
        "nan" if math.isnan(off) else "%+.3f" % off, info["view_ok"], info["reason"],
        info["lap_var"], info["bright"], info["thresh"], info["degrade"])


# --------------------------------------------------------------------------- modes

def run_test(p):
    if p.image:
        frame = cv2.imread(os.path.expanduser(p.image))
        if frame is None:
            raise SystemExit("[camera_node] could not read image %s" % p.image)
        if frame.shape[:2] != (HEIGHT, WIDTH):
            frame = cv2.resize(frame, (WIDTH, HEIGHT))
    else:
        cam = RealSenseCamera()
        try:
            cam.warmup()
            frame = cam.read(5000)
        finally:
            cam.stop()
        if frame is None:
            raise SystemExit("[camera_node] no color frame received")
        if p.save_raw:
            save_debug(frame, os.path.join(os.path.dirname(p.debug_path), "raw.jpg"))

    frame, info, contour, centroid, top = process(frame, p, p.degrade)
    save_debug(draw_debug(frame, info, contour, centroid, top), p.debug_path)
    print(summary(info))
    print("debug image saved to %s" % p.debug_path)


IMAGE_TOPIC = "/perception/debug_image/compressed"


def publish_debug_image(node, pub, img, p):
    """JPEG-encode the annotated frame and publish it as sensor_msgs/CompressedImage."""
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
    from std_msgs.msg import Bool, Float32

    rclpy.init()
    node = Node("camera_node")
    pub_off = node.create_publisher(Float32, "/perception/line_offset", 10)
    pub_ok = node.create_publisher(Bool, "/perception/view_ok", 10)
    pub_img = None
    if p.image_hz > 0:
        from rclpy.qos import QoSProfile, ReliabilityPolicy
        from sensor_msgs.msg import CompressedImage
        pub_img = node.create_publisher(CompressedImage, IMAGE_TOPIC,
                                        QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT))
    last_img = 0.0
    log = node.get_logger()

    cam = RealSenseCamera()
    log.info("camera started %dx%d@%d, line=%s thresh=%s blur_min=%.0f bright=[%.0f,%.0f] degrade=%d ramp=%s"
             % (WIDTH, HEIGHT, FPS, p.line, p.thresh, p.blur_min, p.bright_min, p.bright_max,
                p.degrade, p.ramp))

    t0 = time.monotonic()
    last_debug = 0.0
    last_log = 0.0
    try:
        while rclpy.ok():
            try:
                frame = cam.read(1000)
            except RuntimeError as e:  # frame timeout: tell drivers we can't see
                log.warn("camera read failed: %s" % e)
                frame = None
            now = time.monotonic()

            if frame is None:
                pub_off.publish(Float32(data=float("nan")))
                pub_ok.publish(Bool(data=False))
                rclpy.spin_once(node, timeout_sec=0.0)
                continue

            steps = p.degrade
            if p.ramp > 0 and p.degrade > 0:
                steps = int((now - t0) // p.ramp) % (p.degrade + 1)

            frame, info, contour, centroid, top = process(frame, p, steps)
            pub_off.publish(Float32(data=float(info["offset"])))
            pub_ok.publish(Bool(data=bool(info["view_ok"])))

            if (pub_img is not None and now - last_img >= 1.0 / p.image_hz
                    and node.count_subscribers(IMAGE_TOPIC) > 0):
                last_img = now
                publish_debug_image(node, pub_img, draw_debug(frame, info, contour, centroid, top), p)

            if p.debug_every >= 0 and now - last_debug >= p.debug_every:
                save_debug(draw_debug(frame, info, contour, centroid, top), p.debug_path)
                last_debug = now
            if now - last_log >= 1.0:
                log.info(summary(info))
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
    ap.add_argument("--save-raw", action="store_true", help="(with --test) also save the raw frame as raw.jpg")
    ap.add_argument("--line", choices=["dark", "light"], default="dark", help="tape color vs floor")
    ap.add_argument("--thresh", type=int, default=None,
                    help="gray threshold 0-255 for tape; -1 = Otsu auto (default 90 dark / 170 light)")
    ap.add_argument("--roi-top", type=float, default=0.5,
                    help="only search below this fraction of image height (default 0.5 = bottom half)")
    ap.add_argument("--min-area", type=float, default=400, help="min tape blob area in px")
    ap.add_argument("--max-area-frac", type=float, default=0.6,
                    help="reject blobs bigger than this fraction of the ROI (floor/shadow, not tape)")
    ap.add_argument("--blur-min", type=float, default=60.0, help="Laplacian variance below this = blurry")
    ap.add_argument("--bright-min", type=float, default=40.0, help="mean gray below this = too dark")
    ap.add_argument("--bright-max", type=float, default=230.0, help="mean gray above this = washed out")
    ap.add_argument("--degrade", type=int, default=0, help="artificial blur steps (0 = off)")
    ap.add_argument("--ramp", type=float, default=0.0,
                    help="(ROS mode) cycle degrade 0..N, advancing one step every RAMP seconds")
    ap.add_argument("--debug-path", default=DEFAULT_DEBUG_PATH)
    ap.add_argument("--image-hz", type=float, default=10.0,
                    help="(ROS mode) max rate of %s for the dashboard; 0 = off" % IMAGE_TOPIC)
    ap.add_argument("--image-width", type=int, default=640, help="downscale the dashboard image to this width")
    ap.add_argument("--image-quality", type=int, default=70, help="JPEG quality 1-100 for the dashboard image")
    ap.add_argument("--debug-every", type=float, default=1.0,
                    help="(ROS mode) seconds between debug.jpg saves; -1 = never")
    # ignore ROS args (e.g. --ros-args ...) if someone passes them
    p, _ = ap.parse_known_args(argv)
    if p.thresh is None:
        p.thresh = 90 if p.line == "dark" else 170
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
