#!/usr/bin/env bash
# FloodBot bring-up: one tmux session, one window per node.
#
#   bash run.sh start     preflight checks, then launch everything (drive stays DISARMED)
#   ARM=1 bash run.sh start   same, but drive_node gets --arm  (WHEELS OFF THE GROUND first time)
#   bash run.sh attach    attach to the session (tmux prefix on this Pi is Ctrl-a; Ctrl-a d detaches)
#   bash run.sh stop      Ctrl-C drive first (neutral + disarm), then everything else, then kill session
#
# Windows: drive | cam | auto | ctl | mon
#
# Overrides (environment variables):
#   DRIVE_DIR CAM_DIR CTL_DIR      default ~/robo-drive ~/robo-cam ~/robo-ctl
#   DRIVE_ARGS CAM_ARGS AUTO_ARGS  extra args, e.g. DRIVE_ARGS="--steer-sign -1 --port /dev/ttyACM1"
#   CTL_NODES                      space-separated control scripts (default: every $CTL_DIR/*_node.py)
#   ARM=1                          pass --arm to drive_node

set -u

SESSION=floodbot
ROS_SETUP=/opt/ros/humble/setup.bash
DRIVE_DIR=${DRIVE_DIR:-$HOME/robo-drive}
CAM_DIR=${CAM_DIR:-$HOME/robo-cam}
CTL_DIR=${CTL_DIR:-$HOME/robo-ctl}
DRIVE_ARGS=${DRIVE_ARGS:-}
CAM_ARGS=${CAM_ARGS:-}
AUTO_ARGS=${AUTO_ARGS:-}
ARM=${ARM:-0}

die()  { echo "run.sh: ERROR: $*" >&2; exit 1; }
warn() { echo "run.sh: warning: $*" >&2; }

# Run a command in a tmux pane with ROS sourced. The pane drops back to a shell
# when the node exits, so crash output stays on screen.
pane_run() {  # target, command
  tmux send-keys -t "$1" "source $ROS_SETUP && $2" C-m
}

preflight() {
  command -v tmux >/dev/null || die "tmux not installed (sudo apt install tmux)"
  [ -f "$ROS_SETUP" ] || die "$ROS_SETUP not found"
  tmux has-session -t "$SESSION" 2>/dev/null && die "session '$SESSION' already running: bash run.sh attach | stop"

  for f in "$DRIVE_DIR/drive_node.py" "$DRIVE_DIR/auto_node.py" "$CAM_DIR/camera_node.py"; do
    [ -f "$f" ] || die "missing $f"
  done

  # Only one process may own the Pixhawk serial port.
  if pgrep -af 'rover_node|mavproxy|drive_node\.py' | grep -v pgrep >/dev/null; then
    pgrep -af 'rover_node|mavproxy|drive_node\.py' | grep -v pgrep >&2
    die "something already talks to the Pixhawk (above). Stop it first (e.g. pkill -f rover_node)."
  fi
  local found=0
  for dev in /dev/ttyACM* /dev/serial/by-id/*; do
    [ -e "$dev" ] || continue
    case "$dev" in /dev/ttyACM*) found=1 ;; esac
    if command -v fuser >/dev/null && fuser "$dev" >/dev/null 2>&1; then
      die "$dev is held by another process: fuser -v $dev"
    fi
  done
  [ "$found" = 1 ] || warn "no /dev/ttyACM* device: Pixhawk not connected? drive_node will fail."
  [ -e /dev/ttyUSB0 ] || warn "/dev/ttyUSB0 (RPLidar) not present."
  if pgrep -af 'realsense|camera_node\.py' | grep -v pgrep >/dev/null; then
    warn "something may already hold the RealSense: $(pgrep -af 'realsense|camera_node\.py' | grep -v pgrep | head -1)"
  fi
}

start() {
  preflight

  local drive_args="$DRIVE_ARGS"
  if [ "$ARM" = 1 ]; then
    drive_args="--arm $drive_args"
    echo "run.sh: ARM=1 -> drive_node will ARM. Wheels off the ground for bench tests!"
  else
    echo "run.sh: drive_node stays DISARMED (use ARM=1 bash run.sh start to arm)."
  fi

  # drive first: it streams neutral before anything else publishes
  tmux new-session -d -s "$SESSION" -n drive -c "$DRIVE_DIR"
  pane_run "$SESSION:drive" "python3 $DRIVE_DIR/drive_node.py $drive_args"

  tmux new-window -t "$SESSION" -n cam -c "$CAM_DIR"
  pane_run "$SESSION:cam" "python3 $CAM_DIR/camera_node.py $CAM_ARGS"

  tmux new-window -t "$SESSION" -n auto -c "$DRIVE_DIR"
  pane_run "$SESSION:auto" "python3 $DRIVE_DIR/auto_node.py $AUTO_ARGS"

  # control nodes (Person C). Not written yet -> placeholder.
  local ctl_nodes="${CTL_NODES:-$(ls "$CTL_DIR"/*_node.py 2>/dev/null | tr '\n' ' ')}"
  tmux new-window -t "$SESSION" -n ctl -c "$CTL_DIR"
  if [ -z "${ctl_nodes// /}" ]; then
    tmux send-keys -t "$SESSION:ctl" "echo 'No control nodes in $CTL_DIR (set CTL_NODES=...). Nothing publishes /cmd/drive, so drive stays neutral.'" C-m
  else
    local first=1
    for n in $ctl_nodes; do
      [ "$first" = 1 ] || tmux split-window -t "$SESSION:ctl" -c "$CTL_DIR"
      pane_run "$SESSION:ctl" "python3 $n"
      first=0
    done
    tmux select-layout -t "$SESSION:ctl" tiled >/dev/null
  fi

  tmux new-window -t "$SESSION" -n mon -c "$DRIVE_DIR"
  pane_run "$SESSION:mon" "ros2 topic echo /cmd/drive"
  tmux split-window -t "$SESSION:mon" -h -c "$DRIVE_DIR"
  pane_run "$SESSION:mon" "ros2 topic echo /cmd/auto"

  tmux select-window -t "$SESSION:drive"
  echo "run.sh: started session '$SESSION' (windows: drive cam auto ctl mon)."
  echo "        attach: bash run.sh attach    stop: bash run.sh stop"
}

stop() {
  tmux has-session -t "$SESSION" 2>/dev/null || { echo "run.sh: no session '$SESSION'"; return 0; }

  # 1. drive first: its Ctrl-C handler sends neutral and disarms.
  echo "run.sh: stopping drive_node (neutral + disarm) ..."
  tmux send-keys -t "$SESSION:drive" C-c
  for _ in $(seq 1 50); do   # up to 5 s (disarm confirmation can take ~3 s)
    pgrep -f 'drive_node\.py' >/dev/null || break
    sleep 0.1
  done
  if pgrep -f 'drive_node\.py' >/dev/null; then
    warn "drive_node still running after 5 s, sending SIGTERM"
    pkill -TERM -f 'drive_node\.py'
    sleep 2
  fi

  # 2. everything else
  for pane in $(tmux list-panes -s -t "$SESSION" -F '#{session_name}:#{window_name}.#{pane_index}'); do
    tmux send-keys -t "$pane" C-c
  done
  sleep 1.5
  tmux kill-session -t "$SESSION"
  echo "run.sh: stopped."
}

case "${1:-start}" in
  start)  start ;;
  stop)   stop ;;
  attach) tmux attach -t "$SESSION" ;;
  *) echo "usage: bash run.sh [start|stop|attach]" >&2; exit 2 ;;
esac
