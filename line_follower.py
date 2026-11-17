import threading, numpy as np, cv2, pyrealsense2 as rs, rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from http.server import BaseHTTPRequestHandler, HTTPServer

# ---------- tunables (near-white / light tape on gray floor) ----------
BRIGHT     = 175     # pixels brighter than this = tape. raise if floor glows green, lower if tape drops out
MIN_AREA   = 400     # ignore bright specks smaller than this (carpet threads)
KP         = 1.2
BASE_SPEED = 0.25
PUBLISH    = True   # preview only until it tracks cleanly
# ----------------------------------------------------------------------
latest = {'img': None}

class LineFollower(Node):
    def __init__(self):
        super().__init__('line_follower')
        self.pub = self.create_publisher(Twist, '/cmd/drive', 10)
        self.pipe = rs.pipeline(); cfg = rs.config()
        cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        self.pipe.start(cfg)
        self.create_timer(0.05, self.tick)

    def tick(self):
        img = np.asanyarray(self.pipe.wait_for_frames().get_color_frame().get_data())
        h, w = img.shape[:2]; y0 = int(h*0.6)
        roi = img[y0:, :]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, BRIGHT, 255, cv2.THRESH_BINARY)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((7,7), np.uint8))
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        t = Twist(); off = None
        if cnts:
            c = max(cnts, key=cv2.contourArea)
            if cv2.contourArea(c) > MIN_AREA:
                M = cv2.moments(c); cx = int(M['m10']/M['m00'])
                off = (cx - w/2)/(w/2)
                t.linear.x = BASE_SPEED
                t.angular.z = max(-1.5, min(1.5, -KP*off))
                cv2.drawContours(roi, [c], -1, (0,255,0), 2)
                cv2.circle(img, (cx, y0+20), 8, (0,0,255), -1)
        if off is None:
            t.linear.x = 0.0; t.angular.z = 0.0
        cv2.line(img, (w//2,0), (w//2,h), (255,0,0), 1)
        g = int(gray[gray.shape[0]//2, w//2])
        label = f"off={off:+.2f}" if off is not None else "LINE LOST -> STOP"
        cv2.putText(img, f"{label} thr={t.linear.x:.2f} steer={t.angular.z:+.2f}",
                    (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
        cv2.putText(img, f"center gray = {g}", (10,60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2)
        latest['img'] = img
        if PUBLISH: self.pub.publish(t)

    def stop(self): self.pub.publish(Twist())

class H(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header('Content-Type','multipart/x-mixed-replace; boundary=f'); self.end_headers()
        try:
            while True:
                img = latest['img']
                if img is not None:
                    self.wfile.write(b'--f\r\nContent-Type: image/jpeg\r\n\r\n'+cv2.imencode('.jpg',img)[1].tobytes()+b'\r\n')
        except Exception: pass
    def log_message(self,*a): pass

def main():
    rclpy.init(); n = LineFollower()
    threading.Thread(target=lambda: HTTPServer(('0.0.0.0',8000),H).serve_forever(), daemon=True).start()
    print('preview http://<PI-IP>:8000')
    try: rclpy.spin(n)
    except KeyboardInterrupt: pass
    finally: n.stop(); n.destroy_node(); rclpy.shutdown()

main()
