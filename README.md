# F1TENTH Catalunya simulator and Follow-the-Gap controller

This repository contains a ROS 2 bridge for the F1TENTH Gym and a reactive Follow-the-Gap controller for the Catalunya track. The controller reads LiDAR and wheel speed to drive, and uses odometry for lap timing and the RViz display.

## Requirements

- Ubuntu 22.04 with ROS 2 Humble installed and sourced
- Python 3.10, `colcon`, and `rosdep`
- The [F1TENTH Gym](https://github.com/f1tenth/f1tenth_gym) Python package
- A graphical desktop session for RViz

Keep the Gym and ROS packages available to the same Python interpreter. The steps below use a virtual environment that can also see ROS's system Python packages.

## 1. Install dependencies

Get this workspace if it is not already checked out:

```bash
cd ~
git clone https://github.com/widegonz/F1Tenth-Repository.git
```

The remaining commands assume the checkout is `~/F1Tenth-Repository`.

Install the ROS build tools and initialize rosdep if this machine has not used it before:

```bash
sudo apt update
sudo apt install -y python3-colcon-common-extensions python3-rosdep python3-venv python3-pip
sudo rosdep init  # Skip this if rosdep is already initialized
rosdep update
```

Create a Python environment, clone the Gym, and install it:

```bash
python3 -m venv --system-site-packages ~/venvs/f1tenth
source ~/venvs/f1tenth/bin/activate
python -m pip install --upgrade pip
git clone https://github.com/f1tenth/f1tenth_gym.git ~/f1tenth_gym
python -m pip install -e ~/f1tenth_gym
```

If `~/f1tenth_gym` already exists, skip the `git clone` command. Install this workspace's ROS dependencies and build the packages:

```bash
source /opt/ros/humble/setup.bash
cd ~/F1Tenth-Repository
rosdep install --from-paths src --ignore-src --rosdistro humble -r -y
colcon build --symlink-install
```

If you cloned this project into a different directory, use that directory instead of `~/F1Tenth-Repository` below.

## 2. Start the simulator

In **Terminal 1**, source ROS, activate the same Python environment, and source the built workspace:

```bash
source /opt/ros/humble/setup.bash
source ~/venvs/f1tenth/bin/activate
cd ~/F1Tenth-Repository
source install/setup.bash
ros2 launch f1tenth_gym_ros gym_bridge_launch.py
```

This starts the Gym bridge, map server, robot transforms, and RViz. Catalunya RViz is loaded by default. To use a different RViz layout, pass its path:

```bash
ros2 launch f1tenth_gym_ros gym_bridge_launch.py rviz_config:=/path/to/other.rviz
```

The launch file uses the Catalunya map packaged with this project if the configured `map_path` cannot be found. The default spawn is `(x=0, y=0, yaw=-2.14363)`; yaw is in radians. Autonomous Follow-the-Gap control is the default. The bridge's `/cmd_vel` teleop input is disabled because its commands override the controller's `/drive` commands. To enable keyboard teleop intentionally, set `kb_teleop: true` in `src/f1tenth_gym_ros/config/sim.yaml`, then rebuild and source the workspace again. The same rebuild step applies when changing the map, spawn, or LiDAR range.

## 3. Start the controller

In **Terminal 2**, use the same environment and workspace setup:

```bash
source /opt/ros/humble/setup.bash
source ~/venvs/f1tenth/bin/activate
cd ~/F1Tenth-Repository
source install/setup.bash
python3 follow_the_gap.py
```

The default `learn` mode starts driving, counts laps, and adjusts its aggression after lap results. The controller publishes `/drive`, reads `/scan` and `/ego_racecar/odom`, and starts from the configured Catalunya pose. Run only one controller for the ego car. If teleop is enabled, do not run a `/cmd_vel` keyboard publisher at the same time: the bridge gives its teleop callback the last word over steering and speed.

The controller's maximum requested speed is **12 m/s**, not a target it holds everywhere. It slows for limited free-space distance, braking distance, and corner curvature. The controller and bridge now both limit LiDAR to 12 m; with the configured 6 m/s² braking and 0.35 m stop distance, a clear 12 m scan horizon can request about 11.6 m/s. The Gym car also physically clips steering at about 0.419 rad even though the configured request is 0.42 rad. Check `ftg_debug.csv`: `cmd_speed` is the controller request, `speed` is the measured Gym speed, and `limiter` identifies what is holding the request down. These files are written in the directory where you start the controller, so start it from the repository root as shown above.

Other controller modes:

```bash
# Use the best clean aggression saved by learn mode, with a safety margin
python3 follow_the_gap.py --ros-args -p mode:=race

# Hold a chosen aggression value fixed
python3 follow_the_gap.py --ros-args -p mode:=fixed -p aggr:=1.0
```

Use decimals for float parameters, such as `1.0`. Press **Ctrl+C** in the controller terminal to stop it cleanly and write the lap log. Keep the simulator running while the controller is active.

## 4. What to expect

- RViz shows the map, LiDAR, car transforms, start/finish line, trail, and race HUD.
- The controller terminal reports lap times, best time, collisions, and aggression.
- `lap_log.csv` stores lap results; `ftg_debug.csv` stores controller diagnostics.
- `learned_params.json` stores learned speed tuning, and `track_memory.json` stores local trouble-spot adjustments.

These files are written relative to the directory where you start `follow_the_gap.py`. Back them up if you want to keep a tuning run; learning mode updates its parameter and memory files.

## Troubleshooting

| Symptom | Check |
|---|---|
| `Package 'f1tenth_gym_ros' not found` | Source `/opt/ros/humble/setup.bash` and `install/setup.bash` in that terminal. Rebuild if needed. |
| `No module named f110_gym` | Activate `~/venvs/f1tenth` and install the F1TENTH Gym package into it. |
| Gym map load error | Confirm the Catalunya map YAML and PNG are present under `src/f1tenth_gym_ros/maps/`; rebuild after adding package assets. |
| RViz does not show the map | Check that map server started, RViz fixed frame is `map`, and `/map` is publishing (`ros2 topic echo /map --once`). |
| Controller waits for data | Check that `/scan` and `/ego_racecar/odom` are publishing (`ros2 topic hz /scan`). |
| Speed seems capped near 7 m/s | Check `cmd_speed` and `limiter` in `ftg_debug.csv`. A `profile`, `brake`, `lateral`, `straight_wall`, or `anticipate` limiter means the controller is deliberately requesting less than 12 m/s for the current path. If `cmd_speed` is higher than the measured `speed`, allow for Gym acceleration or check the bridge's requested command with `ros2 topic echo /drive`. |
| Steering does not match the controller | Keep `kb_teleop: false` for autonomous runs and check for other `/cmd_vel` publishers with `ros2 topic info -v /cmd_vel`; an enabled teleop subscriber overrides `/drive` with its own steering command. Also make sure only one controller publishes `/drive` (`ros2 topic info -v /drive`). |
| Car does not move | Confirm the controller is running, no second node is publishing commands, and `/drive` has a subscriber (`ros2 topic info /drive`). |

## Project layout

```text
follow_the_gap.py                 Catalunya Follow-the-Gap controller and lap logger
hud.py                            Race HUD renderer
src/f1tenth_gym_ros/              Gym bridge, launch file, maps, and RViz layouts
src/controllers/                  ROS controller examples from the laboratory exercises
src/path_planning/                Waypoint recording utility
```

The lab controller files are examples and are not needed to run the Catalunya Follow-the-Gap controller.
# ROS2_F1Tenth
