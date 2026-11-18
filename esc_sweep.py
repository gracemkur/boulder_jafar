#!/usr/bin/env python3
"""
ESC / steering sweep: find which throttle signal makes THIS car go forward, backward, or stop.

    WHEELS OFF THE GROUND. Hand on the car's power switch. Stop drive_node / run.sh first.
    cd ~/robo && python3 esc_sweep.py

Steps through raw MANUAL_CONTROL values (the same numbers drive_node logs as z= and y=),
one step per Enter. At each step, type what the wheels do:
    f = forward   b = backward   s = stopped      (steering steps: l = left, r = right, c = center)
Ctrl+C at any time: neutral, disarm, exit. A summary is printed at the end.
"""
import sys
import threading
import time

from drive_node import Pixhawk, find_port

RATE_HZ = 20.0
STEPS = [  # (label, y = steer, z = throttle)
    ("stop", 0, 0),
    ("throttle z=-100", 0, -100),
    ("throttle z=-200", 0, -200),
    ("throttle z=-300", 0, -300),
    ("stop", 0, 0),
    ("throttle z=+100", 0, 100),
    ("throttle z=+200", 0, 200),
    ("throttle z=+300", 0, 300),
    ("stop", 0, 0),
    ("steer y=+1000", 1000, 0),
    ("steer y=-1000", -1000, 0),
    ("steer center", 0, 0),
]


def main():
    port = find_port(sys.argv[1] if len(sys.argv) > 1 else "auto")
    px = Pixhawk(port)
    px.connect(115200)
    c = px.conn
    # ask for SERVO_OUTPUT_RAW (id 36) at 5 Hz so we can print the real pulse widths
    c.mav.command_long_send(c.target_system, c.target_component, 511, 0, 36, 200000, 0, 0, 0, 0, 0)

    if not px.set_mode("MANUAL"):
        print("could not set MANUAL mode")
        px.close()
        return

    cmd = [0, 0]
    pwm = {}
    stop = threading.Event()

    def pump():
        last_hb = 0.0
        while not stop.is_set():
            px.send_control(cmd[0], cmd[1])
            now = time.monotonic()
            if now - last_hb > 1.0:
                px.send_gcs_heartbeat()
                last_hb = now
            while True:
                msg = c.recv_msg()
                if msg is None:
                    break
                t = msg.get_type()
                if t == "HEARTBEAT":
                    px._on_heartbeat(msg)
                elif t == "SERVO_OUTPUT_RAW":
                    pwm["steer"] = msg.servo1_raw
                    pwm["throttle"] = msg.servo3_raw
                elif t == "STATUSTEXT":
                    print("  [pixhawk] %s" % msg.text)
            time.sleep(1.0 / RATE_HZ)

    th = threading.Thread(target=pump, daemon=True)
    th.start()
    results = []
    try:
        time.sleep(1.0)
        px.send_arm(True)
        t_end = time.monotonic() + 5.0
        while not px.armed and time.monotonic() < t_end:
            time.sleep(0.1)
        print("armed=%s mode=%s" % (px.armed, px.mode))

        print("\nWHEELS OFF THE GROUND. Ctrl+C = stop.\n")
        for i, (label, y, z) in enumerate(STEPS, 1):
            cmd[0], cmd[1] = y, z
            time.sleep(1.5)
            ans = input("[%2d/%d] %-16s -> pulse throttle=%s steer=%s us | wheels? (f/b/s or l/r/c): "
                        % (i, len(STEPS), label, pwm.get("throttle", "?"), pwm.get("steer", "?"))).strip()
            results.append((label, pwm.get("throttle"), pwm.get("steer"), ans))
    except (KeyboardInterrupt, EOFError):
        print("\nstopping ...")
    finally:
        cmd[0], cmd[1] = 0, 0
        time.sleep(0.5)
        stop.set()
        th.join(timeout=2)
        for _ in range(5):
            px.send_control(0, 0)
            time.sleep(0.05)
        px.send_arm(False)
        time.sleep(1.0)
        px.close()
        print("neutral sent, disarm requested.")

    if results:
        print("\n%-16s %10s %8s  %s" % ("step", "throttle", "steer", "you saw"))
        for label, thr, st, ans in results:
            print("%-16s %10s %8s  %s" % (label, thr, st, ans))
        print("\nPaste this table to whoever is tuning drive_node.")


if __name__ == "__main__":
    main()
